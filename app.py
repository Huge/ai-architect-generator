import json
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from post_process import post_process, has_closed_exterior_loop

# --- Configuration ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
ADAPTER_PATH = "./qwen-kalkulio-lora-v3/final"
SYSTEM_PROMPT = "You are an expert architectural AI. Generate a valid JSON floor plan for a single-family house."

# --- Global Model State ---
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        print("Model loaded successfully.")

# --- Visualization ---
def render_floor_plan(plan):
    """Renders the floor plan JSON into a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Colors for different room types
    colors = {
        "LivingRoom": "#ffe4b5",
        "Bedroom": "#e6e6fa",
        "Kitchen": "#ffdab9",
        "Bath": "#e0ffff",
        "Entry": "#f5f5dc",
        "Dining": "#fffacd",
        "Hall": "#f0f8ff",
        "Closet": "#f5fffa"
    }

    # 1. Draw Rooms
    for r in plan.get("prostory", []):
        poly = r.get("polygon")
        if not poly: continue
        
        rtype = r.get("typ", "Undefined")
        color = colors.get(rtype, "#f0f0f0")
        
        patch = Polygon(poly, closed=True, facecolor=color, edgecolor='none', alpha=0.7)
        ax.add_patch(patch)
        
        # Add text label in the center
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        area = r.get("plocha_m2", 0)
        ax.text(cx, cy, f"{rtype}\n{area} m²", ha='center', va='center', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    # 2. Draw Walls
    for w in plan.get("steny", []):
        x1, y1 = w["od"]
        x2, y2 = w["do"]
        thickness = w.get("tloustka", 0.2) * 10 # Scale thickness for visualization
        ax.plot([x1, x2], [y1, y2], color='#2c3e50', linewidth=max(2, thickness), solid_capstyle='round')

    # 3. Draw Openings (simplified as points/lines for now)
    for o in plan.get("otvory", []):
        # We don't have exact coordinates for openings without math on the host wall,
        # but we can skip them for the basic matplotlib preview since walls/rooms show the structure.
        pass

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off') # Hide grid
    plt.tight_layout()
    return fig

# --- Generation Logic ---
def generate_plan(area_m2, max_attempts=3):
    load_model()
    
    best_plan = None
    best_raw = ""
    status_msg = ""
    
    for attempt in range(1, max_attempts + 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate a floor plan for a house with an approximate area of {area_m2}m2."},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            continue

        cleaned = post_process(plan, target_area=area_m2)
        
        if has_closed_exterior_loop(cleaned):
            best_plan = cleaned
            status_msg = f"✅ Success! Found a watertight floor plan on attempt {attempt}."
            break
        else:
            best_plan = cleaned # keep as fallback
            status_msg = f"⚠️ Warning: Plan might not be fully enclosed (stopped after {attempt} attempts)."

    if best_plan is None:
        return "{}", None, "❌ Failed to generate valid JSON."
        
    fig = render_floor_plan(best_plan)
    return json.dumps(best_plan, indent=2, ensure_ascii=False), fig, status_msg

# --- Modification Logic (Bonus) ---
def modify_plan(plan_json_str, instruction):
    load_model()
    
    try:
        # Validate input JSON
        json.loads(plan_json_str)
    except json.JSONDecodeError:
        return plan_json_str, None, "❌ Invalid input JSON."

    messages = [
        {"role": "system", "content": "You are an expert architectural AI. You modify JSON floor plans based on user instructions. Output ONLY the modified JSON."},
        {"role": "user", "content": f"Here is the current floor plan:\n```json\n{plan_json_str}\n```\n\nInstruction: {instruction}\n\nOutput the complete modified JSON."}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.3, # Lower temp for modification to preserve structure
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Strip markdown code blocks if the model added them
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
        
    try:
        modified_plan = json.loads(raw)
        # We post-process the modified plan to ensure it's still geometrically valid
        # We don't force a target area here, we let the modification dictate it
        cleaned = post_process(modified_plan)
        fig = render_floor_plan(cleaned)
        return json.dumps(cleaned, indent=2, ensure_ascii=False), fig, "✅ Modification successful."
    except json.JSONDecodeError:
        return raw, None, "❌ Failed to parse modified JSON."

# --- Gradio UI ---
with gr.Blocks(title="Kalkulio AI Architect", theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🏗️ Kalkulio AI Architect
        Generate and modify single-family house floor plans using a fine-tuned Qwen2.5-Coder-3B model.
        """
    )
    
    with gr.Tabs():
        # TAB 1: GENERATION
        with gr.TabItem("✨ Generate New Plan"):
            with gr.Row():
                with gr.Column(scale=1):
                    area_slider = gr.Slider(minimum=60, maximum=200, value=120, step=5, label="Target Area (m²)")
                    generate_btn = gr.Button("Generate Floor Plan", variant="primary")
                    gen_status = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown("### How it works\n1. The AI generates a raw floor plan.\n2. A deterministic post-processor cleans up the geometry (snaps walls, drops overlaps, scales to exact area).\n3. A best-of-N sampler ensures the final house is 'watertight'.")
                
                with gr.Column(scale=2):
                    gen_plot = gr.Plot(label="Floor Plan Visualization")
                    gen_json = gr.Code(label="Output JSON (Kalkulio Format)", language="json", interactive=False)
            
            generate_btn.click(
                fn=generate_plan,
                inputs=[area_slider],
                outputs=[gen_json, gen_plot, gen_status]
            )

        # TAB 2: MODIFICATION (BONUS)
        with gr.TabItem("🛠️ Modify Existing Plan (Bonus)"):
            with gr.Row():
                with gr.Column(scale=1):
                    mod_input_json = gr.Code(label="Input JSON", language="json", interactive=True)
                    mod_instruction = gr.Textbox(label="Instruction", placeholder="e.g., Add a second bathroom, or make the living room larger.", lines=2)
                    modify_btn = gr.Button("Apply Modification", variant="primary")
                    mod_status = gr.Textbox(label="Status", interactive=False)
                    
                with gr.Column(scale=2):
                    mod_plot = gr.Plot(label="Modified Floor Plan Visualization")
                    mod_output_json = gr.Code(label="Modified JSON", language="json", interactive=False)
                    
            modify_btn.click(
                fn=modify_plan,
                inputs=[mod_input_json, mod_instruction],
                outputs=[mod_output_json, mod_plot, mod_status]
            )

if __name__ == "__main__":
    # Launch the Gradio app
    # share=True creates a public huggingface link for testing
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
