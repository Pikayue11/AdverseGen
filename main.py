import PySimpleGUI as sg
import util_gui as gui
import threading
import Thread_control as tc
from backEnd.attacker import ImageAttacker
import numpy as np

# N = 1
# imageArr = np.zeros((N,32,32,3))
path = ''
threads = []
runningFlag = 0

#   button
Check_image = sg.Button('Check image', key='-ci-')
Comfirm_all = sg.Button('Comfirm all', key='-ca-')
Stop = sg.Button('Stop', key='-stop-')
Quit = sg.Button('Quit', key='-quit-')
Clear = sg.Button('Clear', key='-clear-')
Console = sg.Button('Console', key='-console-')

#   text
# title = sg.Text('Our tool box',key='-title-',justification='center')
t1 = sg.Text('Please select network and database', key='-t1-')
t2 = sg.Text('Please choose a jpg/jpeg/png picture in your computer', key='-t2-')
# t3 = sg.Text('Max queries',key='-t3-')
# t4 = sg.Text('recommend: 100000-200000',key='-t4-')
t5 = sg.Text('progress bar', key='-t5-')
t6 = sg.Text('States: free          ', key='-t6-')

#   Combo
database = sg.Combo(['CIFAR-10', 'ImageNet'], default_value='CIFAR-10', change_submits=True,
                    key='-db-')  # change_submits=True 可以监听combo
network = sg.Combo(['ResNet18', 'AlexNet', 'VGG16'], default_value='ResNet18', key='-nw-')
evaluation = sg.Combo(['L0', 'L2', 'L∞', 'SSIM', 'Decision-based'], default_value='SSIM', key='-ev-')
attackMode = sg.Combo(['NonTarget', 'Target'], default_value='NonTarget', key='-am-')

#   image select
i1 = sg.Input(key='-ImagePath-', change_submits=True)
f1 = sg.FileBrowse('choose picture')

#   image
p1 = sg.Image(size=(100, 200), key='-ori_image-')
p2 = sg.Image(size=(100, 200), key='-pert_image-')
p3 = sg.Image(size=(100, 200), key='-adv_image-')
ori_label = sg.Text('', size=(25, 1), key='-ori_label', justification='center')
pert_value = sg.Text('', size=(25, 1), key='-pert_value-',  justification='center')
adv_label = sg.Text('', size=(25, 1), key='-adv_label-',  justification='center')

#   input
queryLimit = sg.InputText(size=(10, 5), key='ql')
#   output
s1 = sg.Output(size=(100, 10), key='-output-', echo_stdout_stderr=True)

#   ProgressBar
pb = sg.ProgressBar(1000, orientation='h', size=(45, 10), key='progressbar')

left_column = [
    # [title],
    [t1],
    [database, network, evaluation, attackMode],
    [t2],
    [i1, f1],
    # [t3,queryLimit],
    [Check_image, Comfirm_all, Stop, Clear, Quit],
    # [s1],
    # [t5],
    # [pb],
    # [t6]
]

right_column = [[p1, p2, p3], [ori_label, pert_value, adv_label]]

layout1 = [[sg.Column(left_column)],
           [sg.HSeparator()],
           [sg.Column(right_column)],
           [sg.HSeparator()],
           [Console],
           [s1],
           [sg.HSeparator()],
           [t6]
           ]

# window1 = sg.Window('Window 1', layout1, size=(800,600))
window1 = sg.Window('Our tool box', layout1, size=(700, 650))
win2_active = False
imageAttacker = ImageAttacker(window1['-db-'].get())
label = 0
flag = True
ori_images = np.array([])

while True:
    event1, values1 = window1.read(timeout=100)


    if event1 == '-console-':
        if flag:
            s1.hide_row()
        else:
            s1.unhide_row()
        flag = not flag

    if event1 == '-ImagePath-':
        ori_new_path = window1['-ImagePath-'].get()
        ori_image = gui.getImage(ori_new_path)
        ori_image_zoom = gui.convert_to_bytes(ori_new_path, (200, 200))
        p1.update(data=ori_image_zoom)


    if event1 in (None, '-quit-'):  # click quit
        for i in threads:
            tc._async_raise(i.ident, SystemExit)
        break

    if event1 in (None, '-stop-'):  # click quit
        for i in threads:
            tc._async_raise(i.ident, SystemExit)
        threads = []
        window1['-t6-'].update('States: free          ')
        window1['-ori_image-'].update()
        window1['-adv_image-'].update()
        print('stopped')

    if event1 in (None, '-clear-'):
        window1['-output-'].update(value='')

    if not win2_active and event1 == '-ci-':  # click check image
        path = window1['-ImagePath-'].get()
        if path == '':
            print('path is empty')
        else:
            II = gui.ImageInfo(window1['-db-'].get())
            ori_newPath = gui.savePng(path, II.width, II.height)
            if ori_newPath == '':
                print('Warning! Please enter a correct image path!')
            else:
                ori_images = gui.getImage(ori_newPath)
                ori_label_id = imageAttacker.get_label(window1['-nw-'].get(), ori_images / 255)
                label = ori_label_id
                ori_label_name = II.labels[int(ori_label_id)]
                print("The label of the original image is", ori_label_name)
                ori_image_zoom = gui.convert_to_bytes(ori_newPath, (200, 200))
                window1['-ori_image-'].update(data=ori_image_zoom)
                window1['-ori_label'].update(f'label: {ori_label_name}')

    if not win2_active and event1 == '-ca-':  # click confirm all

        if window1['-t6-'].get()[8:12] == 'free':
            window1['-t6-'].update('States: running   ')
            t1 = threading.Thread(target=gui.getAdvPath, args=(imageAttacker, ori_images, label, II, window1,))
            threads.append(t1)
            t1.start()
            t2 = threading.Thread(target=gui.updateRunning, args=(window1,))
            threads.append(t2)
            t2.start()
        else:
            print('There is something running, please wait')

window1.close()
