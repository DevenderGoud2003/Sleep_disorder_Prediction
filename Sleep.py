import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

# Global warning suppression
warnings.filterwarnings('ignore')


# ================== DATA PROCESSING ==================
def load_and_preprocess():
    data = pd.read_csv(
        'C:/Users/sri sitarama swami/PycharmProjects/Optimized/dataset/Sleep_health_and_lifestyle_dataset.csv')

    # Advanced feature engineering
    data['BP_Systolic'] = data['Blood Pressure'].apply(lambda x: int(x.split('/')[0]))
    data['BP_Diastolic'] = data['Blood Pressure'].apply(lambda x: int(x.split('/')[1]))
    data['BP_Ratio'] = np.where(data['BP_Diastolic'] != 0,
                                data['BP_Systolic'] / data['BP_Diastolic'],
                                0)

    # Enhanced features
    data['Sleep_Efficiency'] = data['Sleep Duration'] * data['Quality of Sleep'] / (data['Stress Level'] + 1)
    data['Activity_Stress_Ratio'] = data['Physical Activity Level'] / (data['Stress Level'] + 1)
    data['Cardio_Index'] = (data['Heart Rate'] * data['BP_Systolic']) / (data['Daily Steps'] + 1)
    data['BMI_Score'] = data['BMI Category'].map({'Normal': 0, 'Overweight': 1, 'Obese': 2})
    data['Health_Risk'] = data['BMI_Score'] * data['Stress Level'] / (data['Physical Activity Level'] + 1)

    # Label encoding
    le = LabelEncoder()
    data['Sleep Disorder'] = le.fit_transform(data['Sleep Disorder'])

    # Drop unnecessary columns
    to_drop = ['Person ID', 'Occupation', 'Blood Pressure', 'BMI Category']
    return data.drop(to_drop, axis=1), le


def build_preprocessor():
    numeric_features = ['Age', 'Sleep Duration', 'Quality of Sleep',
                        'Physical Activity Level', 'Stress Level', 'Heart Rate',
                        'Daily Steps', 'BP_Systolic', 'BP_Diastolic', 'BP_Ratio',
                        'Sleep_Efficiency', 'Activity_Stress_Ratio', 'Cardio_Index', 'BMI_Score', 'Health_Risk']

    categorical_features = ['Gender']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# ================== MODEL OPTIMIZATION ==================
def get_optimized_models():
    return {
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {
                'knn__n_neighbors': [3, 5, 7],
                'knn__weights': ['distance'],
                'knn__p': [1]
            }
        },
        'svm': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'svm__C': [10, 100],
                'svm__kernel': ['rbf'],
                'svm__gamma': ['scale'],
                'svm__class_weight': ['balanced']
            }
        },
        'dt': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'dt__max_depth': [None],
                'dt__min_samples_split': [2, 5],
                'dt__criterion': ['gini', 'entropy'],
                'dt__class_weight': ['balanced']
            }
        },
        'rf': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'rf__n_estimators': [300],
                'rf__max_depth': [None],
                'rf__min_samples_split': [2, 5],
                'rf__class_weight': ['balanced']
            }
        },
        'xgb': {
            'model': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss'),
            'params': {
                'xgb__n_estimators': [300],
                'xgb__max_depth': [3, 5],
                'xgb__learning_rate': [0.1],
                'xgb__subsample': [0.8, 1.0]
            }
        },
        'lgbm': {
            'model': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            'params': {
                'lgbm__n_estimators': [300],
                'lgbm__max_depth': [5, 7],
                'lgbm__learning_rate': [0.1],
                'lgbm__min_child_samples': [20],
                'lgbm__class_weight': ['balanced']
            }
        },
        'ann': {
            'model': MLPClassifier(random_state=42, early_stopping=True),
            'params': {
                'ann__hidden_layer_sizes': [(100, 50)],
                'ann__alpha': [0.001],
                'ann__learning_rate_init': [0.01],
                'ann__batch_size': [32]
            }
        }
    }


# ================== MAIN EXECUTION ==================
if __name__ == "__main__":
    # Load and preprocess data
    data, label_encoder = load_and_preprocess()
    X = data.drop('Sleep Disorder', axis=1)
    y = data['Sleep Disorder']

    # Create preprocessing pipeline
    preprocessor = build_preprocessor()

    # Split data with stratification (85-15 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Get model configurations
    models_config = get_optimized_models()

    # Store all results
    all_results = []

    # Train and evaluate models
    for name, config in models_config.items():
        print(f"\n=== OPTIMIZING {name.upper()} ===")

        # Create full pipeline
        model_pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=3)),
            (name, config['model'])
        ])

        # Perform grid search with 5-fold CV
        grid = GridSearchCV(
            model_pipe,
            config['params'],
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        # Train model
        grid.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = grid.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Store metrics
        all_results.append({
            'Model': name.upper(),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Best Params': grid.best_params_
        })

        print(f"\n{name.upper()} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

    # Create super stacking ensemble
    print("\n=== CREATING SUPER STACKING ENSEMBLE ===")

    # Get top 4 models
    sorted_results = sorted(all_results, key=lambda x: x['Accuracy'], reverse=True)[:4]
    estimators = [(res['Model'],
                   models_config[res['Model'].lower()]['model'].set_params(
                       **{k.replace(f"{res['Model'].lower()}__", ''): v
                          for k, v in res['Best Params'].items()}))
                  for res in sorted_results]

    # Create stacking pipeline
    stack_pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('stack', StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            n_jobs=-1,
            passthrough=True
        ))
    ])

    # Train and evaluate
    stack_pipe.fit(X_train, y_train)
    y_pred_stack = stack_pipe.predict(X_test)

    # Calculate metrics
    stack_metrics = {
        'Model': 'SUPER STACK',
        'Accuracy': accuracy_score(y_test, y_pred_stack),
        'Precision': precision_score(y_test, y_pred_stack, average='weighted'),
        'Recall': recall_score(y_test, y_pred_stack, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred_stack, average='weighted')
    }
    all_results.append(stack_metrics)

    # Print results
    print("\nSuper Stack Results:")
    for metric, value in stack_metrics.items():
        if metric != 'Model':
            print(f"{metric}: {value:.4f}")

    # Final summary
    print("\n=== FINAL PERFORMANCE SUMMARY ===")
    summary_df = pd.DataFrame(all_results)
    print(summary_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
