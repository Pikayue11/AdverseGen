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
targetSave = -1

#   button
Check_image = sg.Button('Check image', key='-ci-')
Check_target = sg.Button('Check image', key = '-ci2-') # in table2, for target attack in imagenet
Comfirm_all = sg.Button('Comfirm all', key='-ca-')
Stop = sg.Button('Stop', key='-stop-')
Quit = sg.Button('Quit', key='-quit-')
Clear = sg.Button('Clear', key='-clear-')
Console = sg.Button('Console', key='-console-')

#   text
# title = sg.Text('Our tool box',key='-title-',justification='center')
t1 = sg.Text('Please select database, network model and attack mode', key='-t1-')
t2 = sg.Text('Please choose a jpg/jpeg/png picture in your computer', key='-t2-')
# t3 = sg.Text('Max queries',key='-t3-')
# t4 = sg.Text('recommend: 100000-200000',key='-t4-')
t5 = sg.Text('progress bar', key='-t5-')
t6 = sg.Text('States: free          ', key='-t6-')
t7 = sg.Text('Please select the attack norm and constraints', key='-t7-')

#   Combo
databaseDeflaut = 'CIFAR-10'
# database = sg.Combo(['CIFAR-10', 'ImageNet'], default_value=databaseDeflaut, change_submits=True,key='-db-')  # change_submits=True 可以监听combo
network = sg.Combo(['ResNet18', 'AlexNet', 'VGG16'], default_value='ResNet18', key='-nw-') # in table1
network2 = sg.Combo(['ResNet18', 'AlexNet', 'VGG16'], default_value='ResNet18', key='-nw2-') # in table2
evaluation = sg.Combo(['L0', 'L2', 'L∞', 'SSIM', 'Decision-based'], default_value='SSIM', key='-ev-')
attackMode = sg.Combo(['Target', 'NonTarget'], default_value='Target', change_submits=True, key='-am-')
attackMode2 = sg.Combo(['Target', 'NonTarget'], default_value='Target', change_submits=True, key='-am2-')
labelArr = sg.Combo(values=[''],change_submits=True, key='-la-')

#   image select
i1 = sg.Input(key='-ImagePath_net-', change_submits=True) # target image (ImageNet)
f1 = sg.FileBrowse( 'choose picture', change_submits=True, key='-cp-')

i2 = sg.Input(key='-ImagePath-', change_submits=True) # original image
f2 = sg.FileBrowse('choose picture', key='-cp2-')

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

# tables
#cifar-10
tab1_layout = [[network, attackMode, labelArr]]
#imageNet
tab2_layout = [[network2, attackMode2],[f1, Check_target,i1]]

tab1 = sg.Tab('CIFAR-10', tab1_layout, key='CIFAR-10')
tab2 = sg.Tab('ImageNet', tab2_layout, key='ImageNet')
gro = sg.TabGroup([[tab1,tab2]], enable_events=True, key='-group-', tab_background_color='grey')


layout_top = [[t1],[gro]]

layout_mid = [
    [t7],
    [evaluation],
    [t2],
    [i2, f2],
    [Check_image, Comfirm_all, Stop, Clear, Quit],
]

layout_bottom = [[p1, p2, p3], [ori_label, pert_value, adv_label]]

layout1 = [[sg.Column(layout_top)],
           [sg.HSeparator()],
           [sg.Column(layout_mid)],
           [sg.HSeparator()],
           [sg.Column(layout_bottom)],
           [sg.HSeparator()],
           [Console],
           [s1],
           [sg.HSeparator()],
           [t6]
           ]

# window1 = sg.Window('Window 1', layout1, size=(800,600))
window1 = sg.Window('Our tool box', layout1, size=(700, 650))
win2_active = False
imageAttacker = ImageAttacker('CIFAR-10')
label = 0
flag = True
ori_images = np.array([])

II = gui.ImageInfo(databaseDeflaut)

while True:
    event1, values1 = window1.read(timeout=100)

    if event1 == '-console-':
        if flag:
            s1.hide_row()
        else:
            s1.unhide_row()
        flag = not flag

    if event1 == '-group-':    # select database in table group, window1['-group-'].get() returns 'CIFAR-10' or 'ImageNet'
        II = gui.ImageInfo(window1['-group-'].get())
        arr = II.getLabelArray()
        window1['-la-'].update(values=arr, value=arr[0])

    if event1 == '-am-':    #   select attack mode in table 1
        if window1['-am-'].get() == 'Target':
            arr = II.getLabelArray()
            window1['-la-'].update(values = arr, value= arr[0], visible = True)
        else:
            window1['-la-'].update(visible = False)

    if event1 == '-am2-':  # select attack mode in table 2
        boo = True
        if window1['-am2-'].get() == 'Target':
            boo = False
        else:
            targetSave = -1
        window1['-cp-'].update(disabled=boo)
        window1['-ci2-'].update(disabled=boo)
        window1['-ImagePath_net-'].update(value='', visible=not boo)

    if event1 == '-ci2-':
        path = window1['-ImagePath_net-'].get()
        if path == '':
            print('path is empty')
        else:
            II = gui.ImageInfo(window1['-group-'].get())
            ori_newPath = gui.savePng(path, II.width, II.height)
            if ori_newPath == '':
                print('Warning! Please enter a correct image path!')
            else:
                ori_images = gui.getImage(ori_newPath)
                tar_label_id = imageAttacker.get_label(window1['-nw-'].get(), ori_images / 255)
                targetSave = tar_label_id
                ori_label_name = II.mapLabel(tar_label_id)
                print("The label of the original image is", ori_label_name)
                ori_image_zoom = gui.convert_to_bytes(ori_newPath, (200, 200))
                window1['-ori_image-'].update(data=ori_image_zoom)
                window1['-ori_label'].update(f'label: {ori_label_name}')




    if event1 == '-ImagePath-': # the common part, path of original image
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

    if event1 in (None, '-clear-'):     # click clear
        window1['-output-'].update(value='')

    if event1 == '-ci-':  # click check image
        path = window1['-ImagePath-'].get()
        if path == '':
            print('path is empty')
        else:
            II = gui.ImageInfo(window1['-group-'].get())
            ori_newPath = gui.savePng(path, II.width, II.height)
            if ori_newPath == '':
                print('Warning! Please enter a correct image path!')
            else:
                ori_images = gui.getImage(ori_newPath)
                ori_label_id = imageAttacker.get_label(window1['-nw-'].get(), ori_images / 255)
                label = ori_label_id
                ori_label_name = II.mapLabel(ori_label_id)
                print("The label of the original image is", ori_label_name)
                ori_image_zoom = gui.convert_to_bytes(ori_newPath, (200, 200))
                window1['-ori_image-'].update(data=ori_image_zoom)
                window1['-ori_label'].update(f'label: {ori_label_name}')

    if event1 == '-ca-':  # click confirm all
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
