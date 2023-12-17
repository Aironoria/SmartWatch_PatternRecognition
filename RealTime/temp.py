
import shutil
import os
import pandas as pd
if __name__ == '__main__':
    root ="support/p1_lhl_2"
    for surface in os.listdir(root):
        for gesture in os.listdir(os.path.join(root,surface)):
            for filename in os.listdir(os.path.join(root,surface,gesture)):
                df = pd.read_csv(os.path.join(root,surface,gesture,filename))
                if len(df) == 240:
                    # crop to 200 in middle
                    df = df.iloc[20:220]
                    df.to_csv(os.path.join(root,surface,gesture,filename),index=False)
