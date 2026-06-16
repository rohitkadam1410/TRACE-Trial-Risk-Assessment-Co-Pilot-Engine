import sys

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace Tabs with Row and Sidebar
new_layout = """
        with gr.Row(elem_id="main_container"):
            # ── Sidebar Navigation ──
            with gr.Column(scale=1, elem_id="sidebar_nav"):
                btn_scorer = gr.Button("🎯 Protocol Risk Scorer", elem_classes=["sidebar-btn", "active"])
                btn_copilot = gr.Button("🤖 Protocol Co-Pilot", elem_classes=["sidebar-btn"])
                btn_amd = gr.Button("🚀 AMD Performance", elem_classes=["sidebar-btn"])

            # ── Main Content Area ──
            with gr.Column(scale=4, elem_id="main_content"):
"""
content = content.replace('        with gr.Tabs() as tabs:', new_layout)

# 2. Replace Tabs with Groups
content = content.replace('with gr.Tab("🎯 Protocol Risk Scorer", id="tab_scorer"):', 'with gr.Group(visible=True, elem_id="panel_scorer") as panel_scorer:')
# Replace copilot tab (handles newlines due to black formatting)
content = content.replace('with gr.Tab(\n                "🤖 Protocol Co-Pilot", id="tab_copilot"\n            ) as copilot_tab:', 'with gr.Group(visible=False, elem_id="panel_copilot") as panel_copilot:')
content = content.replace('with gr.Tab("🚀 AMD Performance", id="tab_amd"):', 'with gr.Group(visible=False, elem_id="panel_amd") as panel_amd:')

# 3. Inject Button Wiring at the end, right before Footer
wiring = """
                # ── Wire Sidebar Navigation ──
                def switch_to_scorer():
                    return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), 
                            gr.update(elem_classes=["sidebar-btn", "active"]), 
                            gr.update(elem_classes=["sidebar-btn"]), 
                            gr.update(elem_classes=["sidebar-btn"]))
                
                def switch_to_copilot():
                    return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                            gr.update(elem_classes=["sidebar-btn"]), 
                            gr.update(elem_classes=["sidebar-btn", "active"]), 
                            gr.update(elem_classes=["sidebar-btn"]))

                def switch_to_amd():
                    return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                            gr.update(elem_classes=["sidebar-btn"]), 
                            gr.update(elem_classes=["sidebar-btn"]), 
                            gr.update(elem_classes=["sidebar-btn", "active"]))

                sidebar_outputs = [panel_scorer, panel_copilot, panel_amd, btn_scorer, btn_copilot, btn_amd]

                btn_scorer.click(fn=switch_to_scorer, inputs=[], outputs=sidebar_outputs)
                btn_copilot.click(fn=switch_to_copilot, inputs=[], outputs=sidebar_outputs).then(
                    fn=on_copilot_tab_selected, inputs=[app_state], outputs=[section_dd]
                )
                btn_amd.click(fn=switch_to_amd, inputs=[], outputs=sidebar_outputs)

        # ── Footer ──"""

content = content.replace('        # ── Footer ──', wiring)

# 4. Remove the old copilot_tab.select wiring
old_copilot_wiring = """                copilot_tab.select(
                    fn=on_copilot_tab_selected,
                    inputs=[app_state],
                    outputs=[section_dd],
                )"""
content = content.replace(old_copilot_wiring, '')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Layout rewrite completed successfully.')
