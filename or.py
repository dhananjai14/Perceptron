from utils.all_utils import prepare_data, save_plot
import pandas as pd
from utils.model import Perceptron

def main(data, modelName, plotName, eta, epochs):
    df_OR = pd.DataFrame(data)
    x, y = prepare_data(df_OR)
    model = Perceptron(eta = eta, epochs = epochs)
    model.fit(x,y)
    _ = model.total_loss()

    # saving the model
    model.save(filename = modelName, model_dir = 'model')

    save_plot(df_OR, model, filename = plotName)



# this will an entry point
if __name__ == "__main__":
    OR = {
        'x1': [0,0,1,1],
        'x2': [0,1,0,1],
        'y' : [0,1,1,1]
    }
    EPOCHS = 10 # No of itterations
    ETA = .1  #ETA or learning rate b/w 0 and 1
    main(data = OR, modelName = 'or.model', plotName = 'or.png',
         eta = ETA, epochs = EPOCHS)


