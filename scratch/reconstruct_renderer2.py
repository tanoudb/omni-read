import json, re

lines = {}
for file_path in [
    'C:/Users/tanou/.gemini/antigravity/brain/80da90c7-f461-42d9-9a72-d7c30adffb08/.system_generated/logs/transcript_full.jsonl',
    'C:/Users/tanou/.gemini/antigravity/brain/d74acc29-1990-4889-97bc-af7aac78c6c4/.system_generated/logs/transcript_full.jsonl'
]:
    for l in open(file_path, encoding='utf-8'):
        try:
            data = json.loads(l)
            if data.get('type') == 'VIEW_FILE':
                content = data.get('content', '')
                if 'File Path: `file:///A:/omni%20read/core/renderer.py`' in content:
                    for m in re.finditer(r'^(\d+): (.*)$', content, re.MULTILINE):
                        lines[int(m.group(1))] = m.group(2).rstrip('\r')
        except:
            pass

sorted_line_nums = sorted(lines.keys())
head_lines = open('core/renderer.py', encoding='utf-8').read().splitlines()

blocks = []
if sorted_line_nums:
    current_block = [sorted_line_nums[0]]
    for num in sorted_line_nums[1:]:
        if num == current_block[-1] + 1:
            current_block.append(num)
        else:
            blocks.append(current_block)
            current_block = [num]
    blocks.append(current_block)

reconstructed = []
expected_line = 1

def find_in_head(line_content, start_idx=0):
    for i in range(start_idx, len(head_lines)):
        if head_lines[i] == line_content:
            return i
    return -1

def find_anchor_before(block, min_idx=0):
    for i in range(len(block)-1, -1, -1):
        idx = find_in_head(lines[block[i]], min_idx)
        if idx != -1:
            return idx, block[i]
    return -1, -1

def find_anchor_after(block, min_idx=0):
    for i in range(len(block)):
        idx = find_in_head(lines[block[i]], min_idx)
        if idx != -1:
            return idx, block[i]
    return -1, -1

last_head_idx = -1

for idx, block in enumerate(blocks):
    start_line = block[0]
    
    if start_line > expected_line:
        prev_block = blocks[idx - 1] if idx > 0 else []
        head_idx_before, line_num_before = find_anchor_before(prev_block, last_head_idx)
        head_idx_after, line_num_after = find_anchor_after(block, max(0, head_idx_before))
        
        if head_idx_before != -1 and head_idx_after != -1:
            for i in range(head_idx_before + 1, head_idx_after):
                reconstructed.append(head_lines[i])
        else:
            print(f"Could not anchor gap {expected_line} to {start_line-1}")
            # fallback
    
    for num in block:
        reconstructed.append(lines[num])
    
    _, last_line_num = find_anchor_before(block, last_head_idx)
    if last_line_num != -1:
        last_head_idx = find_in_head(lines[last_line_num], last_head_idx)
        
    expected_line = block[-1] + 1

if expected_line <= 3348:
    head_idx_before, _ = find_anchor_before(blocks[-1], last_head_idx)
    if head_idx_before != -1:
        for i in range(head_idx_before + 1, len(head_lines)):
            reconstructed.append(head_lines[i])

open('scratch/renderer_reconstructed.py', 'w', encoding='utf-8').write('\n'.join(reconstructed))
print(f"Reconstructed {len(reconstructed)} lines.")
