

def main():
    # If processed path not found, process EmoSet-118K (train/val/test set)
    # if not os.path.exists('/red/ruogu.fang/share/emotion_adversarial_attack/data/processed/EmoSet-118K'):
        
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, help='Which model to use for training')
    args = parser.parse_args()