from django.shortcuts import render
from django.http import HttpResponseRedirect
import pandas as pd
from joblib import load
import numpy as np
from .forms import InputForm

model = load('models/finalized_model4.sav')
vector = load('models/finalized_model5.sav')


def home(request) :
    return render(request,'twitter/form.html')


def quality(request):
        # if this is a POST request we need to process the form data
        if request.method == 'POST' :
            # create a form instance and populate it with data from the request:
            form = InputForm ( request.POST )
            # check whether it's valid:
            if form.is_valid ( ) :
                # process the data in form.cleaned_data as required
                message = form.cleaned_data [ 'message' ]
                # ...
                value = message
                data = [ value ]
                vect = vector.transform ( data ).toarray ( )
                result = model.predict ( vect )
                if result == 0 :
                    var = 'not spam'
                else :
                    var = 'spam'
                context = {'var':var}
                return render(request,'twitter/result.html',context)

        # if a GET (or any other method) we'll create a blank form
        else :
            form = InputForm( )
        return render ( request, 'twitter/form.html', {'form' : form} )
