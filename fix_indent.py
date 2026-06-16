with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_main_content = False

for line in lines:
    if 'with gr.Column(scale=4, elem_id="main_content"):' in line:
        in_main_content = True
        new_lines.append(line)
        continue
        
    if '# ── Wire Sidebar Navigation ──' in line:
        in_main_content = False
        
    if in_main_content and line.strip() != '':
        new_lines.append('    ' + line)
    else:
        new_lines.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
    
print('Indentation fixed.')
