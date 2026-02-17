import pandas as pd
import os
import shutil
import glob

# ================= CONFIGURATION =================
# 1. List your game folders exactly as they are named
GAME_FOLDERS = ["game_2", "game_4", "game_5", "game_6", "game_7"]

# 2. Define the Output Folder (The one the training code will read)
OUTPUT_DIR = "real_data_pack"
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "real_gt.csv")

# =================================================

def process_games():
    # Create the output directories
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    
    final_data = []
    total_images_processed = 0

    print(f"--- Starting Processing for {len(GAME_FOLDERS)} Games ---")

    for game_name in GAME_FOLDERS:
        print(f"\n📂 Processing {game_name}...")
        
        # 1. Find the CSV file inside the game folder
        csv_path = glob.glob(os.path.join(game_name, "*.csv"))
        if not csv_path:
            print(f"   ⚠️ WARNING: No CSV found in {game_name}. Skipping.")
            continue
        
        # Read the raw CSV
        # Assumes columns: 'fen', 'to_frame', 'from_frame'
        try:
            df = pd.read_csv(csv_path[0])
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            continue

        # 2. Iterate through the CSV rows
        images_in_game = 0
        for index, row in df.iterrows():
            fen = row['fen']
            frame_num = int(row['to_frame'])
            
            # Construct the filename as it appears in the folder (e.g., frame_000200)
            # We check for both .jpg and .png just to be safe
            base_filename = f"frame_{frame_num:06d}" 
            
            # Look for the file in the 'images' subfolder
            src_path = None
            extension = ""
            
            possible_files = [
                os.path.join(game_name, "images", f"{base_filename}.jpg"),
                os.path.join(game_name, "images", f"{base_filename}.png"),
                os.path.join(game_name, "images", f"{base_filename}.jpeg")
            ]
            
            for p in possible_files:
                if os.path.exists(p):
                    src_path = p
                    extension = os.path.splitext(p)[1] # Get .jpg or .png
                    break
            
            # 3. IF FILE EXISTS (Meaning you didn't delete it) -> Process it
            if src_path:
                # Create a unique name: "game_2_frame_000200.jpg"
                new_filename = f"{game_name}_{base_filename}{extension}"
                dst_path = os.path.join(OUTPUT_IMG_DIR, new_filename)
                
                # Copy the image
                shutil.copy2(src_path, dst_path)
                
                # Add to our list for the Master CSV
                # Columns: image_name, FEN, View_specification
                final_data.append([new_filename, fen, "white"])
                
                images_in_game += 1
                total_images_processed += 1
            
            # If src_path is None, it means you deleted the photo manually.
            # The script simply ignores it.

        print(f"   ✅ Collected {images_in_game} valid images.")

    # ================= SAVE MASTER CSV =================
    if final_data:
        # Create DataFrame with the exact headers required
        final_df = pd.DataFrame(final_data, columns=['image_name', 'FEN', 'View_specification'])
        
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 DONE! Processed {total_images_processed} images total.")
        print(f"📁 Output saved to: {OUTPUT_DIR}")
        print(f"📄 Master CSV: {OUTPUT_CSV}")
    else:
        print("\n❌ Error: No images were found. Check your folder names.")

if __name__ == "__main__":
    process_games()