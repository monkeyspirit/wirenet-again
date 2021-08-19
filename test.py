import ember
import lightgbm as lgb  # Gradient Boosting
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import pickle

from ember import PEFeatureExtractor


def main():
    X_train, y_train, X_test, y_test = ember.read_vectorized_features("ember2018")
    model = lgb.Booster(model_file="ember2018/ember_model_2018.txt")
    model.params['objective'] = 'binary'

    explainer = shap.TreeExplainer(model, )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    def plotImp(model, num=40, fig_size=(40, 20)):
        """
        Grafica le 40 feature piu' importanti.
        """
        feature_imp = pd.DataFrame(
            {'Value': model.feature_importance(importance_type='gain'), 'Feature': model.feature_name()})
        plt.figure(figsize=fig_size)
        sns.set(font_scale=2)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                         ascending=False)[0:num])
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances-01.png')
        plt.show()

    plotImp(model)

    shap_values = explainer.shap_values(np.asarray(X_test.tolist())[:100])
    shap.summary_plot(shap_values[0], np.asarray(X_test.tolist())[:100], max_display=20, sort=True)
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shap_values[1][20])
    winrar_data = open("/home/maria/Scaricati/WinRAR-x64-602it.exe", "rb").read()

    extractor = PEFeatureExtractor(2)

    features = np.array(extractor.feature_vector(winrar_data), dtype=np.float32)


if __name__ == '__main__':
    main()
