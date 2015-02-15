from deep.datasets.load import load_plankton
X, y = load_plankton()
X_test, y_test = load_plankton(test=True)

from deep.augmentation import Reshape
size = 28
X = Reshape(size).transform(X)
X_test = Reshape(size).transform(X_test)

import numpy as np
def augment(X):
    X = X.reshape(-1, size, size)
    X_45 = np.rot90(X.T, 1).T.reshape(-1, size**2)
    X_90 = np.rot90(X.T, 2).T.reshape(-1, size**2)
    X_180 = np.rot90(X.T, 3).T.reshape(-1, size**2)

    X_lr = np.fliplr(X)
    X_45_lr = np.rot90(X_lr.T, 1).T.reshape(-1, size**2)
    X_90_lr = np.rot90(X_lr.T, 2).T.reshape(-1, size**2)
    X_180_lr = np.rot90(X_lr.T, 3).T.reshape(-1, size**2)

    X = X.reshape(-1, size**2)
    X_lr = X_lr.reshape(-1, size**2)

    return X, X_45, X_90, X_180, X_lr, X_45_lr, X_90_lr, X_180_lr

X = np.vstack(augment(X))
y = np.hstack((y, y, y, y, y, y, y, y))

from sklearn.preprocessing import StandardScaler
X = np.vstack((X, X_test))
X = StandardScaler().fit_transform(X)
X_test = X[-len(X_test):]
X = X[:-len(X_test)]

from sklearn.cross_validation import train_test_split
X, X_valid, y, y_valid = train_test_split(X, y, test_size=.1)

from deep.datasets import SupervisedData
train = SupervisedData((X, y))
valid = SupervisedData((X_valid, y_valid))

from deep.layers.base import Layer, PreConv, PostConv, Pooling, ConvolutionLayer
from deep.activations import RectifiedLinear, Softmax
layers = [
    PreConv(),
    ConvolutionLayer(64, 7, 1, RectifiedLinear()),
    Pooling(5, 2),
    ConvolutionLayer(64, 5, 1, RectifiedLinear()),
    Pooling(3, 2),
    PostConv(),
    Layer(1000, RectifiedLinear()),
    Layer(121, Softmax())
]

from deep.networks import NN
from deep.updates import Momentum
from deep.fit import EarlyStopping, Iterative
nn = NN(layers, learning_rate=.01, update=Momentum(.9), fit=EarlyStopping(valid=valid, batch_size=128))
nn.fit(train)

from theano import function
predictions = []
for A in augment(X_valid):
    n_batches = len(A) / 100
    valid.X.set_value(A)
    batch_start = valid.batch_index * 100
    batch_end = (valid.batch_index+1) * 100
    givens = {nn.x: valid.X[batch_start:batch_end]}
    prediction_function = function([valid.batch_index], nn._symbolic_predict_proba(nn.x), givens=givens)
    prediction = []
    for i in range(n_batches+1):
        prediction.extend(prediction_function(i))
    predictions.append(prediction)

predictions = np.asarray(predictions)
prediction = np.argmax(np.mean(predictions, axis=0), axis=1)
print np.mean(prediction == y_valid)

predictions = []
for A in augment(X_test):
    n_batches = len(A) / 100
    valid.X.set_value(A)
    batch_start = valid.batch_index * 100
    batch_end = (valid.batch_index+1) * 100
    givens = {nn.x: valid.X[batch_start:batch_end]}
    prediction_function = function([valid.batch_index], nn._symbolic_predict_proba(nn.x), givens=givens)
    prediction = []
    for i in range(n_batches):
        prediction.extend(prediction_function(i))
    predictions.append(prediction)

predictions = np.asarray(predictions)
predictions = np.mean(predictions, axis=0)

with open('test_submission.csv', 'wb') as submission:
    submission.write('image,acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified\n')

    for prediction, y in zip(predictions, y_test):
        line = str(y) + ',' + ','.join([str(format(i, 'f')) for i in prediction]) + '\n'
        submission.write(line)