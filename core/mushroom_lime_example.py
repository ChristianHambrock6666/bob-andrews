import numpy as np
import sklearn.ensemble
import lime.lime_tabular



# ------------------------------------------------------------------------------------------------------
# prepare data: ----------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
data = np.genfromtxt('../data/agaricus-lepiota.data', delimiter=',', dtype='<U20')
labels = data[:, 0]
le = sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:, 1:]

categorical_features = range(22)
feature_names = 'cap-shape,cap-surface,cap-color,bruises?,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring, stalk-surface-below-ring, stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat'.split(',')
categorical_names = '''bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
fibrous=f,grooves=g,scaly=y,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises=t,no=f
almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
attached=a,descending=d,free=f,notched=n
close=c,crowded=w,distant=d
broad=b,narrow=n
black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
enlarging=e,tapering=t
bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
fibrous=f,scaly=y,silky=k,smooth=s
fibrous=f,scaly=y,silky=k,smooth=s
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
partial=p,universal=u
brown=n,orange=o,white=w,yellow=y
none=n,one=o,two=t
cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d'''.split('\n')
for j, names in enumerate(categorical_names):
    values = dict([(x.split('=')[1], x.split('=')[0]) for x in names.split(',')])
    data[:, j] = np.array(list(map(lambda x: values[x], data[:, j])))

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_

print(data[:, 0])
print(categorical_names[0])

data = data.astype(float)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)

encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
encoder.fit(data)
encoded_train = encoder.transform(train)



# ------------------------------------------------------------------------------------------------------
# fit model: -------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(encoded_train, labels_train)



# ------------------------------------------------------------------------------------------------------
# LIME from here: --------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
predict_fn = lambda x: rf.predict_proba(encoder.transform(x))
explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=['edible', 'poisonous'],
                                                   feature_names=feature_names,
                                                   categorical_features=categorical_features,
                                                   categorical_names=categorical_names, kernel_width=3, verbose=False)
i = 127
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.save_to_file("../output/test_html.out")

