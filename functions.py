class feedforwardClassifier(nn.Module):
    def __init__(self):
        super(feedforwardClassifier, self).__init__()
        
        self.fc1 = nn.Linear(in_features=1, out_features=4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=4, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        
        hidden = self.fc1(x)
        activ = self.relu(hidden)
        hidden = self.fc2(activ)
        logit = self.activation(hidden)
        return logit

    
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1),
            nn.ReLU())
        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2))
        
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.activation(x)
        return x

def cluster_texts(texts, clusters=3):
    """ Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
    vectorizer = TfidfVectorizer(tokenizer=process_text,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)
 
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(tfidf_model)
 
    clustering = collections.defaultdict(list)
 
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
 
    return clustering
 

def process_text(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    table = str.maketrans("","",string.punctuation)    
    text = text.translate(table)
    tokens = word_tokenize(text)
 
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
 
    return tokens

def tribonacci_rec(n):
    if n <=2: 
        return 1
    
    res = tribonacci_rec(n-1) + tribonacci_rec(n-2) + tribonacci_rec(n-3)
    return res
    
        
# implement a non recursive (iterative) version here
def tribonacci_norec(n):
    fib_nums = [1, 1, 1]
    
    if n <=2: return fib[:n]
    
    for i in range(3, n):
        curr_fib = fib_nums[i-1] + fib_nums[i-2] + fib_nums[i-3]
        fib_nums.append(curr_fib)
    return fib_nums

tribonacci_rec(5)

def encode(msg, shift):
    car_list = list(msg)
    for i in range(len(msg)):
        if car_list[i].isalpha():
            car_list[i] = chr((ord(msg[i])-ord('a')+shift)%26+ord('a'))
    res = "".join(car_list)
    return res
    

def decode(coded_msg, shift):
    return encode(coded_msg, -1*shift)

encode('a',26)

def find_duplicates(seq):
    count = Counter(seq) #Counter create a dictonnary where values are number of occurence of the key
    unique = []
    repeat = set() # we use set to have unique occurence of repeating number

    for i in range(len(seq)):
        if count[seq[i]] == 1:
            unique.append(seq[i])
        else:
            repeat.add(seq[i])
    
    return {'repeat': repeat, 'unique': sorted(unique)} #we use sorted because Counter doesn't sort by value

def cartesian(a, b):
    k = 0 #should be a list not a int
    for x in range(len(a)):
        for y in range(len(b)):
            k.append(a[x] + ", " + b[x]) # can not append to an int 'k'
                                         # can not use '+' with int and str (we need operands to be of the same type)
                                         # --> we need to convert 'a' into string
                    
        #there is no return in the function, I could not return anything as 'k' is not an argument of the function
cartesian([1,2,3], ["a","b"])

def add(a, b):
    return a+b


def foo():

    return 'hello world' 

def bar():
    return 'Ok USA !'
