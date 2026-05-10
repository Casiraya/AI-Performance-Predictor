from django.shortcuts import render
import joblib
import pandas as pd
import os
from django.conf import settings

# 1. Load the model (Ensure this matches your newly exported joblib file name)
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'student_pass_model1.joblib')
model = joblib.load(MODEL_PATH)

def predict_performance(request):
    prediction = None
    
    if request.method == 'POST':
        # 2. Get the data from the HTML form
        study_hours = float(request.POST.get('study_hours', 0))
        attendance = float(request.POST.get('attendance', 0))
        previous_score = float(request.POST.get('previous_score', 0))
        score_improvement = float(request.POST.get('score_improvement', 0)) 
        parent_ed_hs = int(request.POST.get('parent_education_hs', 0)) 
        
        # 3. Format it in the EXACT order the model was trained on
        input_data = pd.DataFrame([[
            study_hours, 
            attendance, 
            previous_score, 
            score_improvement, 
            parent_ed_hs
        ]], columns=[
            'study_hours_per_week', 
            'attendance_rate', 
            'previous_score', 
            'score_improvement',
            'parent_education_High School'
        ])
        
        # 4. Make the prediction
        result = model.predict(input_data)
        
        if result[0] == 1:
            prediction = "PASS ✅"
        else:
            prediction = "FAIL ❌"

    # 5. Pass both the prediction AND the user's submitted data back to the template
    context = {
        'prediction': prediction,
        'submitted_data': request.POST if request.method == 'POST' else None
    }
    
    return render(request, 'predict.html', context)