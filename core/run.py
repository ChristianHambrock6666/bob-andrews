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

    #evaluator.setup_lime_explainer(sess, loader.get_train_sentence_char_lists())
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

    # ------------------------------------------------------------------------------------------------------
    # LIME from here: --------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------
    feature_names = [str(num) for num in range(200)]
    categorical_features = range(200)
    categorical_names = [['-'] + [char_trf.num2char[ch+1] for ch in range(len(char_trf.num2char))] for num in range(200)]
    train = np.array([np.array(char_trf.tensor_to_numbers(tensor)) for tensor in loader.get_next_train_batch(1000)[0]])

    predict_fn = lambda num_sentences: np.array([
        np.array(evaluator.predict(sess, char_trf.numbers_to_tensor(num_sentence)))
        for num_sentence in num_sentences])

    explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=['yes', 'no'],
                                                   feature_names=feature_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3, verbose=False)

    i = 127
    exp = explainer.explain_instance(train[i], predict_fn, num_features=5)
    exp.save_to_file("../output/testXX_html.out")
    d = None

tex_writer.compile()
subprocess.call(["xdg-open", tex_writer.outputFile])
