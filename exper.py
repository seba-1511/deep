print 'Loading Planktons...'
from deep.datasets.load import load_plankton
X, y = load_plankton()
X_test, y_test = load_plankton(test=True)

from deep.augmentation import Reshape
X = Reshape(48).fit_transform(X)
X_test = Reshape(48).fit_transform(X_test)

import numpy as np
X = np.vstack((X, X_test))

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

print 'Defining Net...'
X_test = X[-len(X_test):]
X = X[:-len(X_test)]

from deep.layers import Layer, PreConv, ConvolutionLayer, Pooling, PostConv
from deep.activations.base import RectifiedLinear, Softmax
from deep.corruptions import Dropout
layers = [
    PreConv(),
    ConvolutionLayer(48, 3, RectifiedLinear()),
    ConvolutionLayer(96, 3, RectifiedLinear(), Dropout(.40)),
    Pooling(3, 3),
    ConvolutionLayer(128, 5, RectifiedLinear(), Dropout(.40)),
    Pooling(3, 2),
    PostConv(),
    Layer(3000, RectifiedLinear(), Dropout(.68)),
    Layer(2500, RectifiedLinear(), Dropout(.68)),
    Layer(121, Softmax(), Dropout(.5))
]

print 'Learning...'
from deep.models import NN
from deep.updates import Momentum
from deep.regularizers import L2
from deep.fit import Iterative
from deep.plot.base import plot_training
nn = NN(layers, .01, Momentum(.9), fit=Iterative(15), regularize=L2(.0005))
nn.fit(X, y)
plot_training(nn)


#: move this to fit
n_batches = len(X_test) / 100

from theano import shared, function
X_test = shared(X_test)

import theano.tensor as T
i = T.lscalar()
x = T.matrix()
batch_start = i * 100
batch_end = (i+1) * 100
givens = {x: X_test[batch_start:batch_end]}
prediction_function = function([i], nn._symbolic_predict_proba(x), givens=givens)

predictions = []
for i in range(n_batches):
    predictions.extend(prediction_function(i))
predictions = np.asarray(predictions)

with open('test_submission.csv', 'wb') as submission:
    submission.write('image,acantharia_protist_big_center,acantharia_protist_halo,acantharia_protist,amphipods,appendicularian_fritillaridae,appendicularian_s_shape,appendicularian_slight_curve,appendicularian_straight,artifacts_edge,artifacts,chaetognath_non_sagitta,chaetognath_other,chaetognath_sagitta,chordate_type1,copepod_calanoid_eggs,copepod_calanoid_eucalanus,copepod_calanoid_flatheads,copepod_calanoid_frillyAntennae,copepod_calanoid_large_side_antennatucked,copepod_calanoid_large,copepod_calanoid_octomoms,copepod_calanoid_small_longantennae,copepod_calanoid,copepod_cyclopoid_copilia,copepod_cyclopoid_oithona_eggs,copepod_cyclopoid_oithona,copepod_other,crustacean_other,ctenophore_cestid,ctenophore_cydippid_no_tentacles,ctenophore_cydippid_tentacles,ctenophore_lobate,decapods,detritus_blob,detritus_filamentous,detritus_other,diatom_chain_string,diatom_chain_tube,echinoderm_larva_pluteus_brittlestar,echinoderm_larva_pluteus_early,echinoderm_larva_pluteus_typeC,echinoderm_larva_pluteus_urchin,echinoderm_larva_seastar_bipinnaria,echinoderm_larva_seastar_brachiolaria,echinoderm_seacucumber_auricularia_larva,echinopluteus,ephyra,euphausiids_young,euphausiids,fecal_pellet,fish_larvae_deep_body,fish_larvae_leptocephali,fish_larvae_medium_body,fish_larvae_myctophids,fish_larvae_thin_body,fish_larvae_very_thin_body,heteropod,hydromedusae_aglaura,hydromedusae_bell_and_tentacles,hydromedusae_h15,hydromedusae_haliscera_small_sideview,hydromedusae_haliscera,hydromedusae_liriope,hydromedusae_narco_dark,hydromedusae_narco_young,hydromedusae_narcomedusae,hydromedusae_other,hydromedusae_partial_dark,hydromedusae_shapeA_sideview_small,hydromedusae_shapeA,hydromedusae_shapeB,hydromedusae_sideview_big,hydromedusae_solmaris,hydromedusae_solmundella,hydromedusae_typeD_bell_and_tentacles,hydromedusae_typeD,hydromedusae_typeE,hydromedusae_typeF,invertebrate_larvae_other_A,invertebrate_larvae_other_B,jellies_tentacles,polychaete,protist_dark_center,protist_fuzzy_olive,protist_noctiluca,protist_other,protist_star,pteropod_butterfly,pteropod_theco_dev_seq,pteropod_triangle,radiolarian_chain,radiolarian_colony,shrimp_caridean,shrimp_sergestidae,shrimp_zoea,shrimp-like_other,siphonophore_calycophoran_abylidae,siphonophore_calycophoran_rocketship_adult,siphonophore_calycophoran_rocketship_young,siphonophore_calycophoran_sphaeronectes_stem,siphonophore_calycophoran_sphaeronectes_young,siphonophore_calycophoran_sphaeronectes,siphonophore_other_parts,siphonophore_partial,siphonophore_physonect_young,siphonophore_physonect,stomatopod,tornaria_acorn_worm_larvae,trichodesmium_bowtie,trichodesmium_multiple,trichodesmium_puff,trichodesmium_tuft,trochophore_larvae,tunicate_doliolid_nurse,tunicate_doliolid,tunicate_partial,tunicate_salp_chains,tunicate_salp,unknown_blobs_and_smudges,unknown_sticks,unknown_unclassified\n')

    for prediction, y in zip(predictions, y_test):
        line = str(y) + ',' + ','.join([str(format(i, 'f')) for i in prediction]) + '\n'
        submission.write(line)
