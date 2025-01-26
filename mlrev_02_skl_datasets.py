# Import necessary libraries
from sklearn import datasets

# clear_data_home()
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels (species)

# datasets.clear_data_home()
ng_data = datasets.fetch_20newsgroups()
ngv_data = datasets.fetch_20newsgroups_vectorized(subset='test')
cal_housing = datasets.fetch_california_housing()
cov_type = datasets.fetch_covtype()
kdd = datasets.fetch_kddcup99()
lfw = datasets.fetch_lfw_pairs()


########################################
# datasets.dump_svmlight_file()
# datasets.fetch_openml()
# datasets.fetch_file()
china = datasets.load_sample_image('china.jpg')

plt.figure(figsize=(8, 6))

# Subplot 1
plt.subplot(4, 1, 1)
plt.imshow(china, label="all 3 channels")
plt.legend()

# Subplot 2
plt.subplot(4, 1, 2)
plt.imshow(china[:, :, 0], label="hannel 1")
plt.legend()

# Subplot 3
plt.subplot(4, 1, 3)  # 3 rows, 1 column, position 3
plt.imshow(china[:, :, 1], label="channel 2")
plt.legend()

# Subplot 4
plt.subplot(4, 1, 4)  # 3 rows, 1 column, position 3
plt.imshow(china[:, :, 2], label="channel 3")
plt.legend()

plt.tight_layout()  # Adjust spacing between plots
plt.show()

########################################
datasets.fetch_20newsgroups()
datasets.fetch_20newsgroups_vectorized()
datasets.fetch_california_housing()
datasets.fetch_covtype()
datasets.fetch_kddcup99()
########################################
datasets.fetch_lfw_pairs()
datasets.fetch_lfw_people()
datasets.fetch_olivetti_faces()
datasets.fetch_rcv1()
datasets.fetch_species_distributions()
########################################
datasets.get_data_home()
datasets.load_breast_cancer()
datasets.load_diabetes()
datasets.load_digits()
datasets.load_files()
########################################
datasets.load_iris()
datasets.load_linnerud()
datasets.load_sample_images()
datasets.load_svmlight_file()
########################################
datasets.load_svmlight_files()
datasets.load_wine()
datasets.make_biclusters()
datasets.make_blobs()
datasets.make_checkerboard()
########################################
datasets.make_circles()
datasets.make_classification()
datasets.make_friedman1()
datasets.make_friedman2()
datasets.make_friedman3()
########################################
datasets.make_gaussian_quantiles()
datasets.make_hastie_10_2()
datasets.make_low_rank_matrix()
datasets.make_moons()
datasets.make_multilabel_classification()
########################################
datasets.make_regression()
datasets.make_s_curve()
datasets.make_sparse_coded_signal()
datasets.make_sparse_spd_matrix()
########################################
datasets.make_sparse_uncorrelated()
datasets.make_spd_matrix()
datasets.make_swiss_roll()
########################################
# datasets.fetch_file()
