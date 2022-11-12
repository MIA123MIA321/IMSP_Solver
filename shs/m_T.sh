#!/bin/bash


N=64
k='40,60,80'
m=16
maxq=0.1
q_method='T'
ARGNAME='m'


LOGDIR='logs/'
OUTPUT_LOG=$LOGDIR'output_'$ARGNAME'_'$q_method'.log'
PYDIR='../'
MAIN_PATH=${PYDIR}'Main.py'
WRITE_PATH=${PYDIR}'Write_J.py'
DRAW_PATH=${PYDIR}'draw.py'
TMP_PATH='.tmp.log'
# Args                    DEFAULT  
# -----------------------------------
# N                       50
# k                       '2'
# m                       3
# maxq	                  0.1
# q0_times                0.0
# q_method                '1'
# noise_level_model       0.0
# noise_level_data        0.0
# gtol                    1e-10
# maxiter                 30
# pic_list                [0,1,2,4,8,-2,-1]
# title                   REQUIRED
# picdir                  'pic/process_jpg/'
# gifdir                  'pic/process_gif/'
# output_filename         REQUIRED
# lamb                    0.0


echo > ${OUTPUT_LOG} &&




m=16 &&
TITLE=$ARGNAME$m'_'$q_method &&
echo > ${TMP_PATH} &&
nohup python -u \
${MAIN_PATH} \
--N ${N} \
--k ${k} \
--m ${m} \
--q_method ${q_method} \
--maxq ${maxq} \
--output_filename ${OUTPUT_LOG} \
--title ${TITLE} \
>> ${TMP_PATH} 2>&1 &&
python -u ${WRITE_PATH} \
--output_filename ${OUTPUT_LOG} &&


m=32 &&
TITLE=$ARGNAME$m'_'$q_method &&
echo > ${TMP_PATH} &&
nohup python -u \
${MAIN_PATH} \
--N ${N} \
--k ${k} \
--m ${m} \
--q_method ${q_method} \
--maxq ${maxq} \
--output_filename ${OUTPUT_LOG} \
--title ${TITLE} \
>> ${TMP_PATH} 2>&1 &&
python -u ${WRITE_PATH} \
--output_filename ${OUTPUT_LOG} &&



m=64 &&
TITLE=$ARGNAME$m'_'$q_method &&
echo > ${TMP_PATH} &&
nohup python -u \
${MAIN_PATH} \
--N ${N} \
--k ${k} \
--m ${m} \
--q_method ${q_method} \
--maxq ${maxq} \
--output_filename ${OUTPUT_LOG} \
--title ${TITLE} \
>> ${TMP_PATH} 2>&1 &&
python -u ${WRITE_PATH} \
--output_filename ${OUTPUT_LOG} &&



m=128 &&
TITLE=$ARGNAME$m'_'$q_method &&
echo > ${TMP_PATH} &&
nohup python -u \
${MAIN_PATH} \
--N ${N} \
--k ${k} \
--m ${m} \
--q_method ${q_method} \
--maxq ${maxq} \
--output_filename ${OUTPUT_LOG} \
--title ${TITLE} \
>> ${TMP_PATH} 2>&1 &&
python -u ${WRITE_PATH} \
--output_filename ${OUTPUT_LOG} &&




# '--savepath', default='pic/res/')
python ${DRAW_PATH} \
--filename ${OUTPUT_LOG} \
--argname ${ARGNAME} \
--q_method ${q_method} &



# tail -f /data/liuziyang/inversepde1/.tmp.log
# 来查看运算进度