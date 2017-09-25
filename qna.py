import numpy
import json
import os.path
import sys
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer # countvectoriser creats tokens for each data set
import numpy.linalg as LA #importing the linear algebra module
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
numpy.seterr(divide='ignore', invalid='ignore')
dict_11_15 = {
    '1':{ 'Why should I consider Analytics career?':'here are a plethora of answers to substantiate the reasons to consider Analytics career. Highly secured job, perfect work-life balance, exceedingly paid salaries, huge job opportunities, vertical career path etc.',
    'What skillset do I need for Analytics career? ':'Strong domain knowledge, Quantitative skills, Business understanding, basic programming skills, creativity, problem solving techniques, other industry specific personality skills and above all you should be an analytics aspirant',
    'Who are the training faculty members?':'We have a team of experienced industry professionals with an average experience of more than 15 years possessing extensive in-depth knowledge in handling analytics tools. They also have a proven expertise on having delivered knowledge and system based training programs who will be working hard to simplify the theoretical concept making them easily understandable.',
    'Who is ECS Datamatics?':'ECS Datamatics is a venture of ECS Inc which is a US based IT services firm established in 2004 and has executed many projects in the government and financial services industry in the past decade. Now, ECS Inc has established a Research Academy (ECS Datamatics) in Chennai offering programs in Data Science, Big Data Analytics, IoT, AI and other emerging fields with paid internship and assured placement support.',
    'How do I start my career in Analytics being fresher?':'ensure your resume grabs the eyeballs of HR of analytics firm, you need to possess all the skillsets of a data engineer where you can acquire this knowledge from a leading analytics training institute like ECS datamatics. Learning Data analytics and implementing your learning practically is the only way being a fresher.',
    'I am a college going student. Can I apply for ECS programs?':'Yes. You can prefer weekend classes but the placement opportunity should be availed only after the completion of your graduation',
    'Is it a paid internship?':'Yes. Few of our programs are backed with paid internship where the stipend will differs from firm to firm',
    'What would be my role in internship?':'You will be assigned with tasks similar to regular employees with a Reporting Authority supervising you throughout the intern period',
    'Can I get places within the Chennai city as I am married young mother and relocation will not be possible?':'Yes. We do understand the demands on a working mother',
    'What types of job profiles are offered to ECS students?':'Depending on the hiring analytics firms, our trainees get various roles like Data Analyst, Big Data Engineer, Machine Learning Engineer, Hadoop Developer, Jr Data Scientist, Big Data Architect, Consultant – Machine Learning, Natural Learning Programming Architect, Statistical Modelling Expert amongst many others',
    'What kind of packages can we expect?':'It all depends on your performance in the interview and how you contribute to the employing organization as there are no limits for career progression. Trainees emerging out most successful in the training will likely to get an average salary of Rs. 3.5 Lacs to Rs. 8 Lacs',
    'I have an experience of 5 years. Will you give me an added advantage for the same in getting in higher salary?':'For the initial job in the Analytics industry, there will not be any difference but it may be helpful for future appraisals. However, the experience you carry will be ascertained as a corporate exposure',
    'Can I pay my fees in installments?':'Yes. Most of the programs have easy installment options; Apart, we do have a tie up with financial institutions for providing loan which can be split and paid in EMIs',
    'I work for six days a week. Is there any possibility of doing the program online?':'Yes. We have two modes of training – Classroom training and online training',
    'Do I need any technical/ programming skills to become a Data Analyst?':'Programming skills are essential but need not be a hardcore programmer to learn Data Science. Having familiarity in any one of the programming skill would ease the learning process',
    'Is there any possibility to arrange for a compensatory class if I miss any of the session?':'Yes. Our program coordinators will assist you in arranging for a quick wrap up session with the faculty else will be getting you the video recording of the session missed',
    'Is the placement is 100 guaranteed?':'No. We do not provide any fake promises to anyone for joining our program. Our unique training methodology will make the trainee a Day 1 Hour 1 productivity professional after the course. We take the complete responsibility of assisting the trainees in placements',
    'Is there a certificate for the program?':'Yes. We provide a proper certificate at the successful completion of the program',
    'Are there any conditions to avail the placement support?':'Yes. You are obliged to maintain 80 of attendance and qualify all the assignments, project works and mock interviews'
    }
}
# lmtzr = WordNetLemmatizer()
# lmtzr_word = lambda list_words: lmtzr.
lmtzr = WordNetLemmatizer()
def model(train_dataset,new_data):
    # new = [str(input())]
    # print(type(new))
    
    new = [new_data]
    ques_list = list(train_dataset.keys())
    lemma_ques_list = [lmtzr.lemmatize(word) for word in ques_list]
    lemma_input_ques= [lmtzr.lemmatize(word) for word in new]
    # print(type(ques_list))
    vectorizer, trainVectorizerArray = train_func(lemma_ques_list)

    new_test = vectorizer.transform(lemma_input_ques).toarray()  # creating a token for the new input data
    # to see what the new token looks like
    # COSINE SIMILARITY algorithm
    # print(new)
    tfidf_transformer = TfidfTransformer()
    new_test1 = tfidf_transformer.fit_transform(new_test).toarray()
    trainVectorizerArray1 = tfidf_transformer.fit_transform(trainVectorizerArray).toarray()
    

    for testV in new_test1:  # selecting the new token that was created for the input question
        cos = 0.0
        ans = ''
        for n, vector in enumerate(trainVectorizerArray1):  # selecting the first token
            cx = lambda a, b: round(numpy.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)
            cosine = cx(vector, testV)  # finding the cosine similarity between the selected token and the new token
            ###########FINDING THE HIGHEST SMILARITY###########3
            if cosine > cos:
                cos = cosine
                # print(type(train_dataset))
                a = ques_list[n]
                ans = train_dataset[a]
        if cos > (0.8):
            return(cos,new_data,ans)
        elif(cos < (0.8) and cos > (0.2)):
        	return(cos,a,ans)
        else:
            return (cos,new_data,'Sorry! I couldn\'t understand that. Be more specific')
            # print("------------------------------")

def train_func(train):
    stopWords = stopwords.words('english')
    # stopWords = ['the', 'is', 'are', 'were', 'a', 'an', 'was', 'has', 'had', 'have','to','do','of','on','my','any','be','by'] #the words that should be ignored by countvectoriser
    vectorizer = CountVectorizer(stop_words=stopWords)  # adding the words list to countvectoriser
    # training data
    train_set = train # creating the training set
    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()  # creating tokens froms the trainng set, This is a 2D array
    # print(trainVectorizerArray)  # just to help us debbug
    return vectorizer,trainVectorizerArray

def main_bot(question_id, user_query):                  # question_id - string, user_query
    question_dict = dict_11_15[question_id]
    cos,a,answer = model(question_dict, user_query)
    if cos > (0.9):
        print('Question:',a,'\nAns:',answer)
    elif(cos < (0.9) and cos > (0.2)):
    	print('I dont have the Ans to your question but I guess i\'ve found something similar\n')
    	print('Question:',a,'\nAns:',answer)
    	yes_no=input('\n\nWas that helpful?\nType:Yes/No\n').lower()
    	if yes_no == 'yes':
    		print('Happy to help!!')
    	else:
    		print('Alright!! I have noted your query, our customer care executive will get back to you')
    else:
        print('Question:',a,'\nAns:',answer)
        # return (answer)

    # return (answer)

while True:
    input12 =input('>>')
    main_bot('1',input12)
