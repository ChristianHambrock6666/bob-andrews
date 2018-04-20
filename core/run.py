import subprocess
import tensorflow as tf
import numpy as np
import sklearn.datasets
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier

import core.loader as ld
import core.trainer as tn
import core.network as nw
import core.config as cf
import core.evaluator as ev
import matplotlib.pyplot as plt
from LaTeXTools.LATEXwriter import LATEXwriter as TeXwriter

import lime
import lime.lime_tabular

# from __future__ import print_function

np.random.seed(1)
#
# iris = sklearn.datasets.load_iris()
# train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)
#
# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
# rf.fit(train, labels_train)
#
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
#             verbose=0, warm_start=False)
#
# sklearn.metrics.accuracy_score(labels_test, rf.predict(test))
#
#
# explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)
#
# i = np.random.randint(0, test.shape[0])
# exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=23, top_labels=23)
#
# print(exp.as_list())

output_map_batch = {"batch_count": [], "accuracy_train": [], "cost_train": []}
output_map_epoch = {"epoch": [], "batch_count": [], "cost_test": [], "accuracy_test": []}

cf = cf.Config()
tex_writer = TeXwriter(".././output", "doc")
tex_writer.addSection("Parameters")
tex_writer.addText(cf.to_tex())
loader = ld.Loader(cf)
char_trf = loader.ct
network = nw.Network(cf)
trainer = tn.Trainer(cf, network)
evaluator = ev.Evaluator(cf, network, char_trf)












#
# data = np.genfromtxt('../data/agaricus-lepiota.data', delimiter=',', dtype='<U20')
# labels = data[:, 0]
# le = sklearn.preprocessing.LabelEncoder()
# le.fit(labels)
# labels = le.transform(labels)
# class_names = le.classes_
# data = data[:, 1:]
#
# categorical_features = range(22)
# feature_names = 'cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat'.split(
#     ',')
#
# categorical_names = '''bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
# fibrous=f,grooves=g,scaly=y,smooth=s
# brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# bruises=t,no=f
# almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# attached=a,descending=d,free=f,notched=n
# close=c,crowded=w,distant=d
# broad=b,narrow=n
# black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# enlarging=e,tapering=t
# bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# fibrous=f,scaly=y,silky=k,smooth=s
# fibrous=f,scaly=y,silky=k,smooth=s
# brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# partial=p,universal=u
# brown=n,orange=o,white=w,yellow=y
# none=n,one=o,two=t
# cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'''.split('\n')
# for j, names in enumerate(categorical_names):
#     values = names.split(',')
#     values = dict([(x.split('=')[1], x.split('=')[0]) for x in values])
#     data[:, j] = np.array(list(map(lambda x: values[x], data[:, j])))
#
# categorical_names = {}
# for feature in categorical_features:
#     le = sklearn.preprocessing.LabelEncoder()
#     le.fit(data[:, feature])
#     data[:, feature] = le.transform(data[:, feature])
#     categorical_names[feature] = le.classes_
#
# print(data[:, 0])
# print(categorical_names[0])
#
# data = data.astype(float)
#
# train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)
#
# encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
# encoder.fit(data)
# encoded_train = encoder.transform(train)
#
# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
# rf.fit(encoded_train, labels_train)
#
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=None, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_split=1e-07, min_samples_leaf=1,
#                        min_samples_split=2, min_weight_fraction_leaf=0.0,
#                        n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)
# predict_fn = lambda x: rf.predict_proba(encoder.transform(x))
# sklearn.metrics.accuracy_score(labels_test, rf.predict(encoder.transform(test)))
#
# np.random.seed(1)
#
# explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=['edible', 'poisonous'],
#                                                    feature_names=feature_names,
#                                                    categorical_features=categorical_features,
#                                                    categorical_names=categorical_names, kernel_width=3, verbose=False)
#
# i = 127
# exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
# exp.save_to_file("test")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while loader.epochs < cf.epochs:
        batch_x, batch_y = loader.get_next_train_batch(cf.batch_size, shuffle=cf.shuffle)
        current_output = trainer.train(sess, batch_x, batch_y)

        output_map_batch["batch_count"].append(loader.batches)
        output_map_batch["accuracy_train"].append(current_output[3])
        output_map_batch["cost_train"].append(current_output[1])

        if loader.new_epoch:
            x, y = loader.get_test_data()
            current_test_output = trainer.test(sess, x, y)

            output_map_epoch["epoch"].append(loader.epochs)
            output_map_epoch["batch_count"].append(loader.batches)
            output_map_epoch["cost_test"].append(current_test_output[0])
            output_map_epoch["accuracy_test"].append(current_test_output[1])

        trainer.print_info_()

    evaluator.setup_lime_explainer(sess, loader.get_train_sentence_char_lists())
    #  ------- information part just for visualization ------------------------------------------------------------
    # plot loss and accuracy
    tex_writer.addSection("Convergence plots")
    fig, ax1 = plt.subplots()
    ax1.plot(output_map_batch["batch_count"], output_map_batch["accuracy_train"])
    ax1.plot(output_map_epoch["batch_count"], output_map_epoch["accuracy_test"])
    ax1.plot(output_map_batch["batch_count"], output_map_batch["cost_train"])
    ax1.plot(output_map_epoch["batch_count"], output_map_epoch["cost_test"])
    plt.xlabel('batch')
    plt.ylabel('cost/accuracy')
    plt.plot()
    tex_writer.addFigure(fig, caption="Accuracy/loss of the training (blue/green) and the test (orange/red) data.")

    # colorize text examples
    tex_writer.addSection("Text examples")
    tex_writer.addText("""The text is colored red if the character was important for the prediction in the following sense:\n\n
    The character is removed (set to default). The prediction is thus changed. 
    The bigger the change towards the category 'no-word-found' of the prediction, the brighter is the character colored. 
    \\vspace{1cm}
    \n\n
    """)
    batch_x, batch_y = loader.get_next_train_batch(cf.batch_size, shuffle=cf.shuffle)
    for tensor_sentence, truth in zip(batch_x, batch_y):
        importance, pred0 = evaluator.importanize_tensor_sentence(sess, tensor_sentence)

        for i in range(len(importance)):
            sentence = char_trf.tensor_to_string(tensor_sentence)
            if not sentence[i] == " ":
                tex_char = "{\color[rgb]{" + str(round(min(importance[i] * 100, 1), 3)) + ",0,0} " + sentence[i] + "}"
            else:
                tex_char = " "
            tex_writer.addText(tex_char)

        tex_writer.addText(",\quad {\\footnotesize $Gray{truth:" + str(round(truth[1], 2)) + ",~pred:~" + str(
            round(pred0[1], 2)) + "}}\n\n")

tex_writer.compile()
subprocess.call(["xdg-open", tex_writer.outputFile])
