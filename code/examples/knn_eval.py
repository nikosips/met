import os
import sys
import pickle
import json
import argparse
import datetime

from code.utils.utils import *
from code.classifiers.knn_classifier import *




def main():
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument('directory', metavar='EXPORT_DIR', help='directory where train descriptors will be loaded from and where results should be saved')
	parser.add_argument('--info_dir', default=None, type=str, help = 'directory where ground truth is stored') 
	parser.add_argument('--mini', action='store_true', help = 'used if evaluating on the mini db')
	parser.add_argument('--dim', '-d', default=512, type=int, help = 'dimensionality after pcaw')
	parser.add_argument('--t', '-t', default=None, type=float, help = 'softmax temperature')
	parser.add_argument('--k', '-k', default=None, type=int, help = 'number of neighbors to consider')
	parser.add_argument('--autotune', action='store_true')
	parser.add_argument('--log', default=None, type=str)
	parser.add_argument('--val_eval', action='store_true', help = 'for evaluating on the val set')
	parser.add_argument('--raw', action='store_true') #use raw descriptors, no PCAw.
	parser.add_argument('--gpuid', default=0, type=int) #id of the gpu in your machine (for faiss)

	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)

	if args.log is None:
		log = open(args.directory+"/log_knn.txt", 'a')

	else:
		log = open(args.directory+"/log_knn_{}.txt".format(args.log), 'a')   
	
	log.write("\n"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n")

	exp_name = "EXP_d{}".format(args.dim)

	if args.autotune:
		exp_name += "AT"
	
	else:
		exp_name += "k{}p{}".format(args.k, args.t)

	if args.val_eval: exp_name += "_val_eval"

	print("expname: {}".format(exp_name))
	log.write("expname: {}\n".format(exp_name))
	print("dir: {}".format(args.directory))
	log.write("dir: {}\n".format(args.directory))

	if args.mini:
			met_db_info_file = args.info_dir + "/mini_MET_database.json"
	else:
		met_db_info_file = args.info_dir + "/MET_database.json"
	
	testset_info_file = args.info_dir + "/testset.json"
	valset_info_file = args.info_dir + "/valset.json"

	#load descriptors
	if os.path.isfile(args.directory+"/descriptors.pkl"):
		print('Loading train descriptors...')
		with open(args.directory+"/descriptors.pkl", 'rb') as data:
			data_dict = pickle.load(data)
			train_descr = np.array(data_dict["train_descriptors"]).astype("float32") 
			test_descr = np.array(data_dict["test_descriptors"]).astype("float32")
			val_descr = np.array(data_dict["val_descriptors"]).astype("float32")

	else:
		sys.exit("File {} does not exist".format(args.directory+"/descriptors.pkl"))

	print("Loaded {} train descriptors!".format(len(train_descr)))

	print("Loaded {} test and {} val descriptors!".format(len(test_descr),len(val_descr)))

	train_descr = np.ascontiguousarray(train_descr, dtype=np.float32)
	test_descr = np.ascontiguousarray(test_descr, dtype=np.float32) 
	val_descr = np.ascontiguousarray(val_descr, dtype=np.float32)


	#load gold labels
	print('Loading ground truth...')

	if os.path.isfile(met_db_info_file):

		with open(met_db_info_file) as data:
			info_list_train = json.load(data)
		
		train_labels = []
		for exhibit in info_list_train:
			train_labels.append(exhibit["id"])
		train_labels = np.array(train_labels)
	
	else:
		sys.exit("File {} does not exist".format(met_db_info_file))

	if os.path.isfile(testset_info_file):
		
		with open(testset_info_file) as data:
			info_list_test = json.load(data)
		
		test_labels = []
		for exhibit in info_list_test:
			try:
				test_labels.append(exhibit["MET_id"])
			except:
				test_labels.append(-1)
		test_labels = np.array(test_labels)
	
	else:
		sys.exit("File {} does not exist".format(testset_info_file))

	if os.path.isfile(valset_info_file):
		
		with open(valset_info_file) as data:
			info_list_val = json.load(data)
		
		val_labels = []
		for exhibit in info_list_val:
			try:
				val_labels.append(exhibit["MET_id"])
			except:
				val_labels.append(-1)
		val_labels = np.array(val_labels)
	
	else:
		sys.exit("File {} does not exist".format(valset_info_file))


	if not args.raw:
		print('Performing post-processing(PCAw) of the descriptors...')
		mean,R = estimate_pca_whiten_with_shrinkage(train_descr,shrinkage=1.0,dimensions=args.dim)

		train_descr = apply_pca_whiten_and_normalize(train_descr,mean,R).astype("float32")
		val_descr = apply_pca_whiten_and_normalize(val_descr,mean,R).astype("float32")
		test_descr = apply_pca_whiten_and_normalize(test_descr,mean,R).astype("float32")


	if args.autotune:
		print('Tuning the K-NN classifier on the validation dataset...')
		if args.k is None:
			params_grid = {'K' : np.array([1,2,3,5,7,10,15,20,50]), 
			't' :np.array([0.01,0.1,1.0,5.0,10.0,15.0,20.0,25.0,30.0,50.0,100.0,500.0])}        
			
		else: 
			params_grid = {'K' : np.array([args.k]), 
			't' :np.array([0.01,0.1,1.0,5.0,10.0,15.0,20.0,25.0,30.0,50.0,100.0,500.0])}
		
		print('Grid of tunable parameters: ' + str(params_grid))

		best_params = tune_KNN(params_grid, train_descr, train_labels, val_descr, val_labels)
		
		print('best parameters are : ' + str(best_params[1]))
		log.write('best parameters are : ' + str(best_params[1])); log.write('\n')
		
		clf = KNN_Classifier(K = int(best_params[1]['K']),t = float(best_params[1]['t']))

	else:

		clf = KNN_Classifier(K = args.k,t = args.t)


	# "train" the knn classifier
	print('Training the classifier...')
	clf.fit(train_descr,train_labels)

	#predict using the knn classifier
	print('Classification on the test set...')

	if args.val_eval:
		test_descr,test_labels = val_descr,val_labels

	test_preds,test_confs= clf.predict(test_descr)


	#report of the results
	print('Evaluation of the test set classification...')
	gap_score,gap_score_non_distr,acc_score = evaluate(test_preds,test_confs,test_labels)
	log.write("GAP {}, GAP-nodis {}, ACC {}\n".format(gap_score,gap_score_non_distr,acc_score));



	
if __name__ == '__main__':
	main()
