import json
import numpy as np

for m in ['perceptron_model_eval.json', 'adaline_model_eval.json', 'hebb_model_eval.json']:
    path = f'backend/models_saved/{m}'
    with open(path) as f:
        d = json.load(f)
    print(m)
    for c in d['test_metrics']['per_class']:
        if c['class'] == 5:
            print(f"  Class 5: accuracy={c.get('accuracy')}, prec={c.get('precision')}, rec={c.get('recall')}, f1={c.get('f1')}")

# Also check confusion matrix row for 5 in perceptron
with open('backend/models_saved/perceptron_model_eval.json') as f:
    d = json.load(f)
cm = np.array(d['test_metrics']['confusion_matrix'])
print("\nPerceptron Confusion row for true 5:")
print("Predicted counts for digits 0..9:", cm[5])
print("Most common prediction for 5:", np.argmax(cm[5]))
