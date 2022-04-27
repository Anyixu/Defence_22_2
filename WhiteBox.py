from secml.data import CDataset
from secml.data.splitter import CDataSplitterKFold
from secml.ml.classifiers import CClassifierSVM
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.peval.metrics import CMetricConfusionMatrix
from secml.adv.attacks.evasion import CAttackEvasionPGD
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Feature_extraction import single_transform
import csv
from statistics import mean, stdev
import threading
import time


def train_test_SVM(x_train_features, x_test_features, y_train, y_test):
    tr_set = CDataset(x_train_features, y_train)
    # Train the SVM
    print("Build SVM")
    xval_splitter = CDataSplitterKFold()
    clf_lin = CClassifierSVM()
    xval_lin_params = {'C': [1]}
    print("Find the best params")
    best_lin_params = clf_lin.estimate_parameters(
        dataset=tr_set,
        parameters=xval_lin_params,
        splitter=xval_splitter,
        metric='accuracy',
        perf_evaluator='xval'
    )
    print("Finish Train")
    print("The best training parameters are: ", [
          (k, best_lin_params[k]) for k in sorted(best_lin_params)])
    print("Train SVM")
    clf_lin.fit(tr_set.X, tr_set.Y)

    # Test the Classifier
    ts_set = CDataset(x_test_features, y_test)
    y_pred = clf_lin.predict(ts_set.X)
    metric = CMetricAccuracy()
    acc = metric.performance_score(y_true=ts_set.Y, y_pred=y_pred)

    confusion_matrix = CMetricConfusionMatrix()
    cm = confusion_matrix.performance_score(y_true=ts_set.Y, y_pred=y_pred)
    print("Accuracy on test set: {:.2%}".format(acc))
    print("Confusion Matrix: ")
    print(cm)
    print("False Positive Rate: {:.2%}".format(39 / (39 + 3445)))
    return tr_set, ts_set, clf_lin


def pdg_attack(clf_lin, tr_set, ts_set, y_test, feature_names, nb_attack, dmax, lb, ub):

    class_to_attack = 1
    cnt = 0  # the number of success adversaril examples

    ori_examples2_x = []
    ori_examples2_y = []

    for i in range(nb_attack):
        # take a point at random being the starting point of the attack
        idx_candidates = np.where(y_test == class_to_attack)
        # select nb_init_pts points randomly in candidates and make them move
        rn = np.random.choice(idx_candidates[0].size, 1)
        x0, y0 = ts_set[idx_candidates[0][rn[0]], :].X, ts_set[idx_candidates[0][rn[0]], :].Y

        x0 = x0.astype(float)
        y0 = y0.astype(int)
        x2 = x0.tondarray()[0]
        y2 = y0.tondarray()[0]

        ori_examples2_x.append(x2)
        ori_examples2_y.append(y2)

    # Perform adversarial attacks
    noise_type = 'l2'  # Type of perturbation 'l1' or 'l2'
    y_target = 0
    # dmax = 0.09  # Maximum perturbation

    # Bounds of the attack space. Can be set to `None` for unbounded

    solver_params = {
        'eta': 0.01,
        'max_iter': 1000,
        'eps': 1e-4}

    # set lower bound and upper bound respectively to 0 and 1 since all features are Boolean
    pgd_attack = CAttackEvasionPGD(
        classifier=clf_lin,
        double_init_ds=tr_set,
        distance=noise_type,
        dmax=dmax,
        lb=lb, ub=ub,
        solver_params=solver_params,
        y_target=y_target
    )

    ad_examples_x = []
    ad_examples_y = []
    ad_index = []
    cnt = 0
    for i in range(len(ori_examples2_x)):
        # print("Current Number:", i)
        x0 = ori_examples2_x[i]
        y0 = ori_examples2_y[i]

        y_pred_pgd, _, adv_ds_pgd, _ = pgd_attack.run(x0, y0)
        # print("Original x0 label: ", y0.item())
        # print("Adversarial example label (PGD): ", y_pred_pgd.item())
        #
        # print("Number of classifier gradient evaluations: {:}"
        #       "".format(pgd_attack.grad_eval))

        if y_pred_pgd.item() == 0:
            cnt = cnt + 1
            ad_index.append(i)

        ad_examples_x.append(adv_ds_pgd.X.tondarray()[0])
        ad_examples_y.append(y_pred_pgd.item())

        attack_pt = adv_ds_pgd.X.tondarray()[0]
    print("PGD attack successful rate:", cnt / nb_attack)
    startTime2 = time.time()
    ori_examples2_x = np.array(ori_examples2_x)
    ori_examples2_y = np.array(ori_examples2_y)
    ad_examples_x = np.array(ad_examples_x)
    ad_examples_y = np.array(ad_examples_y)

    ori_dataframe = pd.DataFrame(ori_examples2_x, columns=feature_names)
    ad_dataframe = pd.DataFrame(ad_examples_x, columns=feature_names)

    # extract the success and fail examples
    ad_dataframe['ad_label'] = ad_examples_y
    ad_success = ad_dataframe.loc[ad_dataframe.ad_label == 0]
    ori_success = ori_dataframe.loc[ad_dataframe.ad_label == 0]
    ad_fail = ad_dataframe.loc[ad_dataframe.ad_label == 1]
    ori_fail = ori_dataframe.loc[ad_dataframe.ad_label == 1]

    ad_success_x = ad_success.drop(columns=['ad_label'])
    ad_fail_x = ad_fail.drop(columns=['ad_label'])

    result = (ad_success_x - ori_success)
    ori_dataframe.to_csv('ori_dataframe.csv')
    ad_dataframe.to_csv('ad_dataframe.csv')
    result.to_csv('result.csv')
    endTime2 = time.time()
    print("after PGD before word14 time is ", endTime2 - startTime2)
    print(ad_success_x)
    print(ori_success)
    print(result)
    return result, cnt, ad_success_x, ori_dataframe, ori_examples2_y


def magical_word(x_train, x_test, y_train, y_test, result, cnt):
    # Method 2
    x2result1 = result
    x2result1 = np.array(x2result1)
    x2result = result
    x2result = x2result.multiply(x2result1)

    sum_number = x2result.sum() / cnt
    sum_number = pd.DataFrame(sum_number, columns=['sum_number'])
    sum_number = sum_number.sort_values(
        by='sum_number', ascending=False, inplace=False)
    print(sum_number)
    sum_number_pd = pd.DataFrame(sum_number.index[:100])
    sum_number_pd.to_csv("x2result.csv")
    d = {'message': x_train, 'label': y_train}
    df = pd.DataFrame(data=d)
    d1 = {'message': x_test, 'label': y_test}
    df1 = pd.DataFrame(data=d1)
    frames = [df, df1]
    messages = pd.concat(frames)
    messages.to_csv("messages.csv")
    spam = messages[messages.label == 1]
    ham = messages[messages.label == 0]

    # Tf-idf for spam datasets
    vect_spam = TfidfVectorizer()
    vect_spam.fit_transform(spam['message'])
    header_spam = vect_spam.get_feature_names()

    # Tf-idf for ham datasets
    vect_ham = TfidfVectorizer()
    vect_ham.fit_transform(ham['message'])
    header_ham = vect_ham.get_feature_names()

    # find unique ham words
    ham_unique = list(set(header_ham).difference(set(header_spam)))
    header_ham1 = pd.DataFrame(ham_unique)
    header_ham1.to_csv("ham_unique.csv")

    with open("x2result.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        top100_features = []
        for row in reader:
            top100_features.append(row[1])
    top100_features = top100_features[1:]
    # in ham & top100

    ham_unique_in_top = list(
        set(ham_unique).intersection(set(top100_features)))
    words14str = ""
    for item in ham_unique_in_top:
        words14str = words14str + " " + item
    return words14str, spam, ham


m2_empty = pd.DataFrame()
spam_cnt = 0
threads = []
m2_empty_l1 = pd.DataFrame()
m2_empty_l2 = pd.DataFrame()
m2_empty_l3 = pd.DataFrame()
m2_empty_l4 = pd.DataFrame()
m2_list = [m2_empty_l1, m2_empty_l2, m2_empty_l3, m2_empty_l4]


class myThread(threading.Thread):

    def __init__(self, threadID, name, spam_message, words14str, method, feature_model, feature_names, scaler, clf_lin, list_index, selection_model):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.spam_message = spam_message
        self.words14str = words14str
        self.method = method
        self.feature_model = feature_model
        self.feature_names = feature_names
        self.scaler = scaler
        self.clf_lin = clf_lin
        self.list_index = list_index
        self.lock = threading.Lock()
        self.selection_model = selection_model

    def run(self):
        global spam_cnt
        print("Starting " + self.name)
        spam_cnt_1 = m2_empty_out(self.name, self.spam_message, self.words14str, self.method,
                                  self.feature_model, self.feature_names, self.scaler, self.clf_lin,
                                  self.list_index, self.selection_model)
        spam_cnt = spam_cnt+spam_cnt_1
        time.sleep(0.1)
        print("Exiting " + self.name)


def m2_empty_out(name, spam_message, words14str, method, feature_model, feature_names, scaler, clf_lin, list_index, selection_model):
    m2_empty_1 = pd.DataFrame()
    spam_cnt_1 = 0
    global m2_list
    #x = 0
    for j in spam_message.message:
        #x = x + 1
        choose_email = [j + words14str]
        message_14_email = pd.DataFrame(choose_email, columns=["message"])
        message_14_tf_idf = single_transform(
            message_14_email["message"], method, feature_model, feature_names, scaler, selection_model)
        message_14_tf_idf = pd.DataFrame(
            message_14_tf_idf.toarray(), columns=feature_names)
        message_14_y = [1]
        message_14_y = pd.Series(message_14_y)
        message_CData = CDataset(message_14_tf_idf, message_14_y)
        message_14_pred = clf_lin.predict(message_CData.X)

        if message_14_pred == 0:
            spam_cnt_1 = spam_cnt_1 + 1
            m2_empty_1 = m2_empty_1.append(
                message_14_tf_idf, ignore_index=True)

    m2_list[list_index] = m2_list[list_index].append(
        m2_empty_1, ignore_index=True)

    #print(m2_list[list_index], file=open("output.txt", "a"))

    return spam_cnt_1


def svm_attack(method, clf_lin, spam, words14str, feature_model, feature_names, scaler, selection_model):

    global m2_empty

    spam_messages = np.array_split(spam, 4)
    print("Start processing message")
    thread1 = myThread(1, "Thread-1", spam_messages[0], words14str,
                       method, feature_model, feature_names, scaler, clf_lin, 0, selection_model)
    thread2 = myThread(2, "Thread-2", spam_messages[1], words14str,
                       method, feature_model, feature_names, scaler, clf_lin, 1, selection_model)
    thread3 = myThread(3, "Thread-3", spam_messages[2], words14str,
                       method, feature_model, feature_names, scaler, clf_lin, 2, selection_model)
    thread4 = myThread(4, "Thread-4", spam_messages[3], words14str,
                       method, feature_model, feature_names, scaler, clf_lin, 3, selection_model)
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)
    threads.append(thread4)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    m2_empty = m2_empty.append(m2_list[0], ignore_index=True)
    m2_empty = m2_empty.append(m2_list[1], ignore_index=True)
    m2_empty = m2_empty.append(m2_list[2], ignore_index=True)
    m2_empty = m2_empty.append(m2_list[3], ignore_index=True)

    print("Exiting Main Thread")
    print('White box attack with length on SVM:')
    print('Number of samples provided:', len(spam))
    print('Number of crafted sample that got misclassified:', spam_cnt)
    print('Successful rate:', spam_cnt / len(spam))

    return m2_empty


def svm_attack_with_l(method, clf_lin, spam, words14str, feature_model, feature_names, scaler, selection_model):
    spam_cnt = 0
    num_thresholds = 5
    len_list = []
    thresholds = []
    m2_empty = pd.DataFrame()
    for j in spam.message:
        len_list.append(len(j))
    gap = max(len_list) - min(len_list)
    for i in range(1, num_thresholds + 1):
        thresholds.append(round((min(len_list) + i * (gap/num_thresholds))))
    print('mean:', mean(len_list))
    print('std:', stdev(len_list))
    print('Max:', max(len_list))
    print('Min:', min(len_list))
    print('thresholds:', thresholds)
    for j in spam.message:
        i = 0
        while len(j) > thresholds[i]:
            i += 1
        times = i + 1
        temp_words14str = ''
        for i in range(times):
            temp_words14str = temp_words14str + words14str
        choose_email = [j + temp_words14str]
        message_14_email = pd.DataFrame(choose_email, columns=["message"])
        message_14_tf_idf = single_transform(
            message_14_email["message"], method, feature_model, feature_names, scaler, selection_model)
        message_14_tf_idf = pd.DataFrame(
            message_14_tf_idf.toarray(), columns=feature_names)
        message_14_y = [1]
        message_14_y = pd.Series(message_14_y)
        message_CData = CDataset(message_14_tf_idf, message_14_y)
        message_14_pred = clf_lin.predict(message_CData.X)

        if message_14_pred == 0:
            spam_cnt = spam_cnt + 1
            m2_empty = m2_empty.append(message_14_tf_idf, ignore_index=True)

    print('White box attack with length on SVM:')
    print('Number of samples provided:', len(spam))
    print('Number of crafted sample that got misclassified:', spam_cnt)
    print('Successful rate:', spam_cnt / len(spam))
    return m2_empty


def whitebox(scaler, feature_model, x_train, x_test, x_train_features, x_test_features, y_train, y_test,
             feature_names, nb_attack, dmax, method, selection_model):

    tr_set, ts_set, clf_lin = train_test_SVM(
        x_train_features, x_test_features, y_train, y_test)
    lb = np.ndarray.min(x_train_features.toarray())
    ub = np.ndarray.max(x_train_features.toarray())
    print(len(feature_names))
    result, cnt, ad_success_x, ori_dataframe, ori_examples2_y = pdg_attack(clf_lin, tr_set, ts_set, y_test,
                                                                           feature_names, nb_attack, dmax, lb, ub)
    startTime1 = time.time()
    words14str, spam, ham = magical_word(
        x_train, x_test, y_train, y_test, result, cnt)
    endTime1 = time.time()
    print("find word14 time is ", endTime1 - startTime1)

    startTime = time.time()
    words14str = "ovalroom och oodah odell oefai ozgen pape overlap oesten pastparticiple outreach ontologically paraconc oneby ozen pedigree pcohen paseo ome paradis ongean oduvm onomatopee overriding passonneau overheard overall oncampus ocke openminded oles olentangy parastoo ouverture omar overt patio oppor outnumber oconnorpe ona pavlidou paralelos peaker outre pellegrin oneself odonnaile omnibus pauwels optalk passe optimal onoyama pejorate overaccept ochsman peg overtone odowd papistes omdat ohayosensei onpremises ociety pejambon oostendorp oma opyt ogot parametrization onomatopoeic offprint ogori pecs peaked onehearer ogasawara ocurred pasch ometime pathbreaking oddly opaque ongeklungen peculiar papier oracio opacity pcl olhn ordonez paulsell pastperfect pastoral parasitic ogino optionality oceanic oote oncall papersubmission pce overovermorgen ordinateur onodera ohagan octobre oft onset ogden oweeshi ojala onuallains onico onzi outspoken oim offertes overly oersnes onedimensional owes orchid ochta oneparagraph olle og okennon pekka overlong oflazer oneplace omoiyari offspring ordre patrikalakis peking pattabhiraman pattern oe par okulska ocracoke ogitools onsite oversees pellegrini ordinance ogico overparsing pbk overmorgen pd ontwikkeld pclx oi ohrid olemiss overlooked onuoy ojoo ognitive pdp oppressing ouwi paradigmes okubo pearson pedagogie orally omniscience papiere patter pauper ogura oneweek onesoon ohta olias pasteuro odonnell ozeki optique ooui paradoxe oleynikova oftthis ohridski paus passus overlapping ohala ofmood oppressed odyssey pecutting patrik pedir ozyurek patras ole optimalite onestop okpewho passive pasteur pauline pawley para peitsara oldfashioned ockhuizen oja omething overthe openly overripe paraphrasing optimistically opposes orator pavia ofelia oreille offset outset onderzoekschool papazachariou oversimplified peerreviewed oitunix paz pcm oviedo oersjoch orden paradigmas ofinstruction odjeljenje oegai onference paradigmbased odt payaqt ontologists optimistic ojitlan operationalized okada overwhelmingly onomasiological ozark oracle ominous ojection ore ohiostate organ optant omtl omdal paraphrased patric overzealous onomatopoeia oversensitivity onuigbo odd oddnumbered orellana oneon pauc oliver pearl opish openaccess odakyu oppen pelcra odlin pckimmo papyrus paradigmatic overridden ouvre oxonian omotic openmid owen pauvre openness passable oppression pdaniels pb pater passer paramo ogourechnikova paucity pavesic offcourse ocus ont odden pedraza onesegment omeara peklund pauci odijk ocr ogrady omission pastor passivized oesterreichische ohannessian pause olfactory onja pawlewski optimising oreilly patriot ochi paramount peech paulus oltmans occurrence outsider ohp omullan peformance octubre onanism oeser overreject oneday onomastics ontogeny pausing oksala officielles oder pcole ordinaire outskirt pear overturned paragraphe patsy optionally optimisation opisi peer peed ock paratactic orchestrating owl pavel pdendale peaceable pearlmutter passiver pcola olsson paranoid ordinarily ov onethird ongo onequarter patrick ontolinguistics opting pea pawlowska oeffentlichkeit pelita occuring oliveira patrizia oncina paucu oded ox papiamentu peiros oit oise pederson paar patton opined ophale opladen ordnung olshansky paperback pasi offhours pedagogique ordzhonikidze peggy paperbark openletter onelittle oclahoma paasonen ochs ojos paans pax onexclusively oculos patruseva paula pdg okazos overstated oficina ozeroff orfqe patriarchal paraphrase onespeaker oxi ondo peacekeeper openended peklenik oneschoolgirl patterson pararradicales oneyear olof odessa olsthoorn odabrani orderly overbooked ocls ohkado pasero offs pathologist ozuka opposing oxy papaz pazienza pdekker ordinary pasta olshausenstr peatfield offshoot oneparty onderzoek opium ollamh oddity pattenrs onwards papua pedersen pavlov pathological onder onno offhand oldest opoudjis paradigm oconnor oldenburg pascual oftnoted omitted oktober pavillon pearance oehrle orales onward oku odense olsen olaf paages officiel ook pathology pascaline pedagogical outof ordinator onboard onederra ove pectacular opted pascale ois okumura pawlak pathologie oclusion pejorative pedia pedrazzini pattimura ozeilla overtaken patternbased overveld paramskas oralidade ocp offputting orchard optimality ofai parag ogihara oceanists pellegrino pastry paun ochanomizu pasttime onyshkevych onelanguage oversight pedagogization op ooyama peacocke olvido ogi payne parallelism omewhere oxfordshire orero pawar paso opportun okiek papuan omain patronage oden odriscoll ordona okay outrage oda onard optimization owing paradoxa ooit ola oj parasesison operative oglau optus ohori ong openbook peirce oversee opt pau optimized oppose oostdijk oot oliva pattiya ovchinnikova oed ohnishi overboard olivier oif oing offerts opportunism parametric pekarek okstate pasttense ojeda overapplication ordboken oppenheim ouvrages pappa papp papergivers oporto pedrycz onepage pavle oystein paterson ordinaires pascal pasquale olling pcfg onderstreepte parabono papyrologists onscreen opposition papen paucal patterning ozemail oleary oviatt pastaza oddsidemargin optimally patti paulo ordinal oceania pdx oguy olive passportsize oni odor oneway pavidis overwhelm payscale ow pdf paulston ontological pasiego ooki ohara ogloblin overtime ogata ohalloran olafur ooi omewhat paradoxical parana orderchange ohme paperno odoru okerson omeone oostende onein palo pedagogic overseeing peculiarity oliphant odder oilfield onesize oeste olga onglides paradox oneinch pappus oy passau oficial onomastic paulin passage odile patron onderwerp oops passivisation paule ovbershort overlapped offsite olbertz paranthetical peeters peansiri overpoort ocell outnumbers ontology oliviero ojeeshi passport oost paarformeln parameterized peircean oneil onufrie oftmentioned oelkers okane oxg oph ops outright paradigmchallenging optimalitytheoretic olchas offglide oxforduniversity oclc parameterization paymnt papazoglou overtly onglide pasture papieren pcmail pedagogy oemboe odu passivization papersstyle peewee parasession orangecounty ographie paraense oxley"
    print(len(words14str))
    m2_empty = svm_attack(method, clf_lin, spam,
                          words14str, feature_model, feature_names, scaler, selection_model)
    endTime = time.time()
    print("used time is ", endTime - startTime)
    # print(m2_empty, file=open("output1.txt", "a"))
    # m2_empty_new = svm_attack_with_l(method, clf_lin, spam, words14str, feature_model, feature_names, scaler)
    return words14str, spam, ad_success_x, m2_empty, tr_set, ts_set, ori_dataframe, ori_examples2_y
