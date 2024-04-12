import os
import pandas as pd
import cv2

def visualize_sequence_images_and_anchors(frames=[1], seq_root='MOT17-01-FRCNN', save_path='results/'):

    # 在seq_root所示的sequence中，按顺序plot出frames所对应的图片，并在图片上画出anchors
    # anchors即为对应gt的锚框
    
    # 先读取gt文件，存储gt文件中的所有目标信息到一个pandas dataframe中
    gt_path = os.path.join(seq_root, 'gt', 'gt.txt')
    df = pd.read_csv(gt_path, header=None)
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'consider_flag', 'class', 'confidence']

    # 读取seqinfo.ini文件，获取图片的长宽
    seqinfo_path = os.path.join(seq_root, 'seqinfo.ini')
    imWidth, imHeight = 0, 0
    with open(seqinfo_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('imWidth'):
                imWidth = int(line.strip().split('=')[1])
            elif line.startswith('imHeight'):
                imHeight = int(line.strip().split('=')[1])

    # 读取图片
    for frame in frames:
        # 图片命名规则为：帧号前面补0直到文件名长度为6，例如第1帧就是000001.jpg，第88帧就是000088.jpg
        image_name = str(frame).zfill(6) + '.jpg'
        image_path = os.path.join(seq_root, 'img1', image_name)
        image = cv2.imread(image_path)

        # 从df中找到frame对应的，所有consider_flag为1的目标
        targets = df[(df['frame'] == frame) & (df['consider_flag'] == 1)]
        for index, target in targets.iterrows():
            x, y, w, h = target['x'], target['y'], target['w'], target['h']
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # 保存图片
        save_image_path = os.path.join(save_path, image_name)
        cv2.imwrite(save_image_path, image)
        print(f'Save image to {save_image_path} successfully!')


if __name__ == '__main__':

    data_root = '/Users/frog_wch/playground/Research/Datasets/MOT17-test-downsample3.0'

    train_sequences = ['MOT17-02-FRCNN']

    for seq in train_sequences:

        seq_root = os.path.join(data_root, 'train', seq)
        frames = [1, 10, 100]
        visualize_sequence_images_and_anchors(frames=frames, seq_root=seq_root, save_path='../results/')
