from django import forms

class InputForm(forms.Form):
     message = forms.CharField(label='Your name', max_length=100)




