import re
import subprocess
import tensorflow as tf
import numpy as np
import os
import core.loader as ld
import core.trainer as tn
import core.network as nw
import core.config as cf
import core.evaluator as ev
from LaTeXTools.LATEXwriter import LATEXwriter as TeXwriter
import lime.lime_text
import lime.lime_tabular
from multiprocessing import Process

cf = cf.Config()


def launchTensorBoard(path):
    import os
    os.system("bash -c \"source /home/chambroc/miniconda3/bin/activate bob && tensorboard --logdir=" + path + "\"")
    return


os.mkdir(cf.tb_dir)
os.mkdir(cf.tb_dir + "/train")
os.mkdir(cf.tb_dir + "/test")
os.mkdir(cf.tb_dir + "/lime")
p = Process(target=launchTensorBoard, args=(cf.tb_dir,))
p.start()
tensorboard_train_writer = tf.summary.FileWriter(cf.tb_dir + "/train")
tensorboard_test_writer = tf.summary.FileWriter(cf.tb_dir + "/test")
tensorboard_lime_writer = tf.summary.FileWriter(cf.tb_dir + "/lime")

tex_writer = TeXwriter(".././output/LaTeX", "doc")
tex_writer.addSection("Parameters")
tex_writer.addText(cf.to_tex())
loader = ld.Loader(cf)
char_trf = loader.ct
network = nw.Network(cf)
trainer = tn.Trainer(cf, network)
evaluator = ev.Evaluator(cf, network, char_trf)
test_features, test_labels = loader.get_test_data()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while loader.epochs < cf.epochs:
        batch_x, batch_y = loader.get_next_train_batch_sample(cf.batch_size)
        current_output = trainer.train(sess, batch_x, batch_y, tensorboard_train_writer)

        if loader.new_epoch:
            current_test_output = trainer.test(sess, test_features, test_labels, tensorboard_test_writer)
            print("epochs: " + str(loader.epochs))

    tex_writer.addSection("Text examples")
    tex_writer.addText("""The text is colored red if the character was important for the prediction in the following sense:\n\n
    The character is removed (set to default). The prediction is thus changed. 
    The bigger the change towards the category 'no-word-found' of the prediction, the brighter is the character colored.\n\\vspace{1cm}\n\n""")

    feature_names = [str(num) for num in range(cf.string_length)]
    categorical_features = range(cf.string_length)
    categorical_names = [['-'] + [char_trf.num2char[ch + 1] for ch in range(len(char_trf.num2char))] for num in
                         range(cf.string_length)]
    predict_text_fn = lambda txt_sentence: np.array(
        evaluator.predict(sess, char_trf.string_to_tensor(char_trf.string_to_const_length(txt_sentence))))
    predict_texts_fn = lambda txt_sentences: np.array([predict_text_fn(txt_sentence) for txt_sentence in txt_sentences])
    explainer_text = lime.lime_text.LimeTextExplainer(split_expression=r'\W+')

    for tensor_sentence, truth in zip(test_features[50:100], test_labels[50:100]):
        sentence = char_trf.tensor_to_string(tensor_sentence).replace("-", " ")
        print(sentence)

        # --------OLD:-------------------------------------------------------
        importance, pred0 = evaluator.importanize_tensor_sentence(sess, tensor_sentence)
        tex_writer.addText("\n\n {\\footnotesize $Gray{truth:" + str(round(truth[1], 2)) + ",~pred:~" + str(
            round(pred0[1], 2)) + "}} (old, lime text)\hrulefill\n\n")
        for i in range(len(importance)):
            if not sentence[i] == " ":
                tex_char = "{\color[rgb]{" + str(round(min(importance[i] * 100, 1), 3)) + ",0,0} " + sentence[i] + "}"
            else:
                tex_char = " "
            tex_writer.addText(tex_char)

        # --------LIME:-------------------------------------------------------
        tex_writer.addText("\n\n")
        word_importances_lime = explainer_text.explain_instance(sentence, predict_texts_fn, num_features=5)
        unique_position_to_importance = dict(word_importances_lime.as_map()[1])
        word_importance_mapping = dict(
            [(word_importances_lime.domain_mapper.indexed_string.as_list[e], unique_position_to_importance.get(i, 0))
             for i, v in enumerate(word_importances_lime.domain_mapper.indexed_string.positions) for e in v])
        word_importance_mapping = {k: v for k, v in word_importance_mapping.items() if v[0] != 5}
        tb_log_text = sess.run(tf.summary.text('word explanation',
                                               tf.convert_to_tensor(
                                            "sentence: \n" + sentence + "\n\nword explainer result:\n" + str(
                                                {k: v for k, v in word_importance_mapping.items() if v[0] != 5}))))
        tensorboard_lime_writer.add_summary(tb_log_text)
        split_sentence = re.split(r'\W+', sentence)
        for word in split_sentence:
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
p.terminate()
subprocess.call(["xdg-open", tex_writer.outputFile])
