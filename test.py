import ember
import lightgbm as lgb  # Gradient Boosting
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import numpy as np
import pickle
import lief

from ember import PEFeatureExtractor


def main():


    X_train, y_train, X_test, y_test = ember.read_vectorized_features("ember2018")
    model = lgb.Booster(model_file="ember2018/ember_model_2018.txt")
    model.params['objective'] = 'binary'

    # explainer = shap.TreeExplainer(model, )

    # warnings.simplefilter(action='ignore', category=FutureWarning)
    #
    # def plotImp(model, num=40, fig_size=(40, 20)):
    #     """
    #     Grafica le 40 feature piu' importanti.
    #     """
    #     feature_imp = pd.DataFrame(
    #         {'Value': model.feature_importance(importance_type='gain'), 'Feature': model.feature_name()})
    #     plt.figure(figsize=fig_size)
    #     sns.set(font_scale=2)
    #     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
    #                                                                      ascending=False)[0:num])
    #     plt.title('LightGBM Features (avg over folds)')
    #     plt.tight_layout()
    #     plt.savefig('lgbm_importances-01.png')
    #     plt.show()
    #
    # print("> Plot the most 40 important features")
    # plotImp(model)

    # print("> Shap values")
    # shap_values = explainer.shap_values(np.asarray(X_test.tolist())[:100])

    # with open('shap_values', 'wb') as s:
    #     pickle.dump(shap_values, s, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('shap_values', 'rb') as handle:
    #     shap_values = pickle.load(handle)
    #
    # print("> Summary plot shap values")
    # shap.summary_plot(shap_values[0], np.asarray(X_test.tolist())[:100], max_display=20, sort=True)


    data = open('output/AddString_5_Tequila.exe', "rb").read()

    extractor = PEFeatureExtractor(2)

    features = np.array(extractor.feature_vector(data), dtype=np.float32)
    with open('output/AddString_5_Tequila', 'wb') as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('features', 'rb') as handle:
    #     features = pickle.load(handle)



def modify_PE():

    def get_binary(file):
        return lief.parse(file)

    def get_builder(binary):
        return lief.PE.Builder(binary)

    def add_section_constant(binary, name, constant, size):
        # create a section
        section = lief.PE.Section(name)

        # fill it with a constant
        section.content = [constant] * size

        # add the section
        binary.add_section(section, lief.PE.SECTION_TYPES.DATA)

        # build and reparse
        builder = get_builder(binary)
        builder.build()
        builder.write('output/AddConstantTequila.exe')
        binary = get_binary('output/AddConstantTequila.exe')

        return binary

    def add_section_strings(binary, name, string_file):
        # grab strings
        with open(string_file, 'r') as fd:
            data = fd.read()

        # create new section
        section = lief.PE.Section(name)

        # convert characters to decimal representation
        section.content = [ord(c) for c in data]

        # add section to binary
        binary.add_section(section, lief.PE.SECTION_TYPES.DATA)

        # build and reparse
        builder = get_builder(binary)
        builder.build()
        builder.write('output/AddString_5_Tequila.exe')
        binary = get_binary('output/AddString_5_Tequila.exe')

        return binary

    def modify_binary(binary):
        header = binary.header
        print(header)
        # print(binary.optional_header)
        print(binary.debug)

    model = lgb.Booster(model_file="ember2018/ember_model_2018.txt")


    # parse internet explorer
    tequila_bin = get_binary('Win32.DarkTequila.exe')

    # add a constant section isn't good => 0.9998018082502094
    # add_section_constant(tequila_bin, "new_section", 250, 500000)

    #1 meh => 0.8737962293318766
    #add_section_strings(tequila_bin, "license", 'netf.txt')
    #2 much better => 0.7946982046366664
    #add_section_strings(tequila_bin, "license", 'winrar.txt')
    #3 very very much better => 0.3025394133707842
    #add_section_strings(tequila_bin, "license", 'notepad.txt')
    #4 meh => 0.7946982046366664
    #add_section_strings(tequila_bin, "license", 'netf+winrar.txt')
    #5 OMG => 0.22589567557279205
    add_section_strings(tequila_bin, "license", 'all.txt')




if __name__ == '__main__':
    main()
    # modify_PE()
