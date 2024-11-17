import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import os

# Function to load data
def classifier_data_loader(filepath):
    data = pd.read_csv(filepath)
    return data

# Function to calculate molecular descriptors
def calculate_descriptors(smiles_list):
    descriptor_names = [
        'MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds', 'TPSA',
        'NumAromaticRings', 'NumAliphaticRings', 'MolMR', 'BalabanJ', 'Chi0v', 'Chi1v',
        'LabuteASA', 'PEOE_VSA1'
    ]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            descriptors.append(calculator.CalcDescriptors(mol))
        else:
            descriptors.append([np.nan] * len(descriptor_names))
            
    return pd.DataFrame(descriptors, columns=descriptor_names)

# Function to train the model
def model_training(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=5)
    auc_scores = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        clf.fit(X_train_fold, y_train_fold)
        if len(np.unique(y_train_fold)) > 1:  # Ensure there are at least two classes
            y_proba = clf.predict_proba(X_test_fold)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_fold, y_proba, pos_label=clf.classes_[1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            auc_scores.append(auc(fpr, tpr))
    
    return clf, tprs, mean_fpr

# Function to output the ROC curve figure
def output_figure(tprs, mean_fpr, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})')

    for i, tpr in enumerate(tprs):
        plt.plot(mean_fpr, tpr, linestyle='--', alpha=0.3)
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest - Five Fold Cross Validation')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'roc_curve.pdf'))

# Function to train and evaluate the classifier
def prior_classifier(filepath):
    # Load and prepare data
    data = classifier_data_loader(filepath)
    descriptor_df = calculate_descriptors(data['smiles'])
    descriptor_df['label'] = data['label']
    descriptor_df = descriptor_df.dropna()
    
    X = descriptor_df.drop('label', axis=1)
    y = descriptor_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model and evaluate
    clf, tprs, mean_fpr = model_training(X, y)
    
    # Output figure
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../eval_classifier')
    output_figure(tprs, mean_fpr, output_dir)
    
    # Train final model and save it
    clf.fit(X_train, y_train)
    joblib.dump(clf, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molecular_classifier.pkl'))

# Function to classify new SMILES
def classify_smiles(smiles):
    descriptors = calculate_descriptors([smiles])
    classifier = joblib.load('molecular_classifier.pkl')
    prediction = classifier.predict(descriptors)
    return prediction[0]

# # If this script is run directly, execute the training and evaluation
# if __name__ == "__main__":
#     prior_classifier('train_NAPro.csv')
