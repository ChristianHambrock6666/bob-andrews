import re
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
import lime.lime_text
import lime.lime_tabular

np.random.seed(1)

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
test_features, test_labels = loader.get_test_data()

with tf.Session() as sess:
    tf.summary.scalar("accuracy", trainer.accuracy)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('../output', sess.graph)

    sess.run(tf.global_variables_initializer())

    while loader.epochs < cf.epochs:
        batch_x, batch_y = loader.get_next_train_batch_sample(cf.batch_size)
        current_output = trainer.train(sess, batch_x, batch_y, merged)
        train_writer.add_summary(current_output[4], loader.batches)

        summary_op = tf.summary.text('tag1', tf.convert_to_tensor('Tag1: Random Text 1' + str(loader.batches)))
        text = sess.run(summary_op)
        train_writer.add_summary(text)

        output_map_batch["batch_count"].append(loader.batches)
        output_map_batch["accuracy_train"].append(current_output[3])
        output_map_batch["cost_train"].append(current_output[1])

        if loader.new_epoch:
            current_test_output = trainer.test(sess, test_features, test_labels)

            output_map_epoch["epoch"].append(loader.epochs)
            output_map_epoch["batch_count"].append(loader.batches)
            output_map_epoch["cost_test"].append(current_test_output[0])
            output_map_epoch["accuracy_test"].append(current_test_output[1])


        trainer.print_info_()

    # evaluator.setup_lime_explainer(sess, loader.get_train_sentence_char_lists())
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
    feature_names = [str(num) for num in range(200)]
    categorical_features = range(200)
    categorical_names = [['-'] + [char_trf.num2char[ch + 1] for ch in range(len(char_trf.num2char))] for num in
                         range(200)]
    train = np.array([np.array(char_trf.tensor_to_numbers(tensor)) for tensor in loader.get_next_train_batch(1000)[0]])

    # --------LIME:-------------------------------------------------------
    predict_fn = lambda num_sentences: np.array([
        np.array(evaluator.predict(sess, char_trf.numbers_to_tensor(num_sentence)))
        for num_sentence in num_sentences])


    def predict_text_fn(txt_sentence):
        ss = txt_sentence
        while len(ss) < 200:
            ss = ss + " "
        if len(ss) > 200:
            print(ss)
            ss = ss[0:200]
        ret = np.array(evaluator.predict(sess, char_trf.string_to_tensor(ss)))
        return ret


    predict_ss_fn = lambda txt_sentences: np.array([predict_text_fn(txt_sentence) for txt_sentence in txt_sentences])
    explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=['absent', 'contained'],
                                                       feature_names=feature_names,
                                                       categorical_features=categorical_features,
                                                       categorical_names=categorical_names, kernel_width=None,
                                                       verbose=False)
    explainer_text = lime.lime_text.LimeTextExplainer(
        split_expression=r'\W+'
        # bow=False  # default True: then words will loose order and double words will be removed as I understand
    )

    for tensor_sentence, truth in zip(test_features[50:100], test_labels[50:100]):
        sentence = char_trf.tensor_to_string(tensor_sentence).replace("-", " ")
        print(sentence)

        # --------OLD:-------------------------------------------------------
        importance, pred0 = evaluator.importanize_tensor_sentence(sess, tensor_sentence)
        tex_writer.addText("\n\n {\\footnotesize $Gray{truth:" + str(round(truth[1], 2)) + ",~pred:~" + str(
            round(pred0[1], 2)) + "}} (old, lime table, lime text)\hrulefill\n\n")
        for i in range(len(importance)):

            if not sentence[i] == " ":
                tex_char = "{\color[rgb]{" + str(round(min(importance[i] * 100, 1), 3)) + ",0,0} " + sentence[i] + "}"
            else:
                tex_char = " "
            tex_writer.addText(tex_char)

        # --------LIME Table:-------------------------------------------------------
        tex_writer.addText("\n\n")
        char_importances_lime = explainer.explain_instance(
            np.array(char_trf.tensor_to_numbers(tensor_sentence)),
            predict_fn, num_features=5)

        dic = dict(char_importances_lime.as_map()[1])
        sum_importance = sum([abs(v) for v in dic.values()])
        for i in range(len(importance)):
            if (not sentence[i] == " ") and (i in dic.keys()):
                if dic[i] > 0:
                    tex_char = "{\color[rgb]{" + str(round(min(dic[i] * 100, 1), 3)) + ",0,0} " + sentence[i] + "}"
                else:
                    tex_char = "{\color[rgb]{0,0," + str(round(min(-dic[i] * 100, 1), 3)) + "} " + sentence[i] + "}"
            else:
                tex_char = sentence[i]
            tex_writer.addText(tex_char)

        # --------LIME Text:-------------------------------------------------------
        tex_writer.addText("\n\n")
        sentence = re.sub(r'\W+', " ", sentence)
        word_importances_lime = explainer_text.explain_instance(
            sentence,
            predict_ss_fn,
            num_features=5)
        dic = dict(word_importances_lime.as_map()[1])
        word_importance_mapping = dict(
            [(word_importances_lime.domain_mapper.indexed_string.as_list[e], dic.get(i, 0)) for i, v in
             enumerate(word_importances_lime.domain_mapper.indexed_string.positions) for e in v])  # this is just weird mapping to unique words
        print(word_importance_mapping)
        sum_importance = sum([abs(v) for v in dic.values()])
        split_sentence = re.split(r'\W+', sentence)
        for idx, word in zip(range(len(split_sentence)), split_sentence):
            word_importance = word_importance_mapping.get(word, 0)
            if word_importance != 0:
                if word_importance > 0:
                    tex_char = "{\color[rgb]{" + str(round(min(word_importance * 100, 1), 3)) + ",0,0} " + word + "}"
                else:
                    tex_char = "{\color[rgb]{0,0," + str(round(min(-word_importance * 100, 1), 3)) + "} " + word + "}"
            else:
                tex_char = word
            if char_trf.contains_pattern(word):
                tex_char = "\\framebox{" + tex_char + "}"
            tex_writer.addText(" " + tex_char)

tex_writer.compile()
subprocess.call(["xdg-open", tex_writer.outputFile])
