import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def get_model_losses(model_folder):
    model2losses = {'train_loss': {}, 'valid_loss': {}}
    for model in os.listdir(model_folder):
        if os.path.isdir(model_folder + '/' + model):
            for k in model2losses:
                model2losses[k][model] = {}
            with open(model_folder + '/' + model + '/training_progress_scores.csv') as fp:
                headers = None
                for line in fp:
                    if headers is None:
                        headers = line.strip().split(',')
                        
                    else:
                        info = line.strip().split(',')
                        for i, h in enumerate(headers[1:]):
                            if info[0] not in model2losses[k][model]:
                                model2losses[k][model]info[0]
                            model2losses[model][info[0]][h] = info[i + 1]
    return model2losses
                        
def plot(model2losses):
    sns.set_theme(style="darkgrid")

    print(model2losses)
    
    df = pd.DataFrame(model2losses)
    print(df)

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', default='models/')
    args = parser.parse_args()

    model2losses = get_model_losses(args.model_folder)
    plot(model2losses)
