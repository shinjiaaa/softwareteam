import os

# 라벨 파일이 들어있는 폴더
label_dir = "Drone Crash Avoidance.v7-v7.yolov11\valid\labels"  # train/labels, val/labels 등 필요에 맞게 수정

# 기존 클래스 -> 새 클래스 번호 매핑
class_map = {0: 2, 1: 3, 2: 4, 3: 1, 4: 0}

# 모든 .txt 파일 순회
for root, dirs, files in os.walk(label_dir):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                old_class = int(parts[0])
                parts[0] = str(class_map[old_class])  # 클래스 번호 변경
                new_lines.append(" ".join(parts) + "\n")
            
            # 덮어쓰기
            with open(path, "w") as f:
                f.writelines(new_lines)

print("클래스 번호 재매핑 완료!")
