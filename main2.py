import PySimpleGUI as sg
from PIL import Image
import os

width = 400
height = 300
path = ''
newPath = ''
checked = False

def savePng(path):
    if os.path.exists(path) == False:
        return ''
    folder_name = 'pngs/'
    file_name = os.path.basename(path)
    if file_name[-3:].lower() == 'jpg':
        file_name = file_name[:-3]
    elif file_name[-4:].lower() == 'jpeg':
        file_name = file_name[:-4]
    elif file_name[-3:].lower() == 'png':
        file_name = file_name[:-3]
    else:
        return ''
    file_name += 'png'
    im = Image.open(path)
    im = im.resize((width,height))
    newPath = folder_name + file_name
    im.save(newPath)
    return newPath



#   button
Check_image = sg.Button('Check image', key='-ci-')
Comfirm_all = sg.Button('Comfirm all', key='-ca-')
Quit = sg.Button('Quit', key='-quit-')
#   text
title = sg.Text('Our tool box',key='-title-',justification='center')
t1 = sg.Text('Please select network and database',key='-t1-')
t2 = sg.Text('Please choose a jpg/jpeg/png picture in your computer',key='-t2-')
t3 = sg.Text('Max queries',key='-t3-')
t4 = sg.Text('recommend: 100000-200000',key='-t4-')
t5 = sg.Text('progress bar',key='-t5-')
#   Combo
database = sg.Combo(['ImageNet','CIFAR-10'],default_value='ImageNet',key='-db-')
network = sg.Combo(['AlexNet','VGG16'],default_value='AlexNet',key='-nw-')
evaluation = sg.Combo(['L0','L2','L∞','SSIM'],default_value='L0',key='-ev-')
attackMode = sg.Combo(['NonTarget','Target'],default_value='NonTarget',key='-am-')
#   image select
i1 = sg.Input(key='-path-')
f1 = sg.FileBrowse('choose picture')
#   image
p1 = sg.Image(size=(width,height),key='-image1-')
p2 = sg.Image(size=(width,height),key='-image2-')
# input
queryLimit = sg.InputText(size=(10,5),key='ql')
# output
output1 = sg.Output(size=(60, 5))
output2 = sg.Output(size=(120, 10))
# , enter_submits=True
#   progressBar 进度条
pb = sg.ProgressBar(1000, orientation='h', size=(40, 10), key='progressbar')


layout1 = [[title],
               [t1],
               [database,network,evaluation,attackMode],
                [t2],
               [i1,f1],
               [t3,queryLimit,t4],
               [Check_image,Comfirm_all,Quit],
               [output1],
               [t5],
               [pb]]

left_image = [[p1]]
right_image = [[p2]]

layout2 = [[sg.Column(left_image),
            sg.VSeperator(),
            sg.Column(right_image)],
           [output2]]

window1 = sg.Window('Our tool box', layout1)
win2_active = False

while True:
    event1, values1 = window1.read(timeout=100)

    if event1 in (None, '-quit-'):
        break

    if not win2_active and event1 == '-ca-' and newPath != '':# comfirm all
        print('here is Comfirm all xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        win2_active = True
        window2 = sg.Window('Warning', layout2,finalize=True)
        window2['-image1-'].update(filename=newPath)
        window2['-image2-'].update(filename=newPath)
        print('this is the result of adversarial attack')

    if win2_active:
        events2, values2 = window2.Read(timeout=100)
        if events2 is None:
            win2_active = False
            window2.close()

    if not win2_active and event1 == '-ci-':# check image
        path = i1.get()
        print('here is check image')
        if(path == ''):
            print('path is empty')
        else:
            newPath = savePng(path)
            if newPath == '':
                print('Warning! Please enter a correct image path!')
            else:
                print('checked, the label of this image is: tiger')




window1.close()