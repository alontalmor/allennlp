
## Version 0.0
#  only takes a command string and runs it on a specific channel, channel is the GPU computer name...
from allennlp.common.util import *
from subprocess import Popen
import traceback,os, psutil
import time
import socket
import shutil
import argparse
import signal
from allennlp.common.elastic_logger import ElasticLogger
import pika

# this will be used for saving to ElasticLogger
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '.')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

connection_params = pika.URLParameters('amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
connection = pika.BlockingConnection(connection_params)
channel = connection.channel()

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("channel", type=str,
                        help="RabbitMQ channel")
parser.add_argument("--shell", type=str, default="not_bash",
                        help="RabbitMQ channel")
args = parser.parse_args()

proc_running = []
iter_count = 0 # counting iteration for writing status
while True:
    try:
        iter_count+=1

        # checking status of all processes
        for proc in proc_running:
            # check if process still alive:
            try:
                print('checcking if process alive '  + str(proc['pid']))
                os.killpg(os.getpgid(proc['pid']), 0)
            except:
                proc['alive'] = False

            log_data = []
            try:
                # Log snapshot
                log_data = proc['log_handle'].readlines()
                proc['log_snapshot'] = ' '.join(log_data)
            except:
                proc['alive'] = False


            # Log time out handling TODO
            try:
                statbuf = os.stat(proc['log_file'])
                proc['log_update_diff'] = time.time() - statbuf.st_mtime
            except:
                print('file not found')
            #proc['log_update_diff'] > 2000:

            if not proc['alive']:
                # checking if process successfully completed task,
                # printing longer log
                if len(log_data) > 100:
                    proc['log_snapshot'] = ' '.join(log_data[-100:])

                # TODO this is an ugly check, but because we are forking with nohup, python does not provide any good alternative...
                if proc['log_snapshot'].find('Traceback (most recent call last):') > -1 or \
                    proc['log_snapshot'].find('error') > -1:
                    ElasticLogger().write_log('INFO', "Job died", flatten_json(proc), push_bulk=True,print_log=True)
                    proc['log_handle'].close()
                    # Requeue
                    #channel.basic_nack(proc['job_tag'])
                    proc_running.remove(proc)
                    break
                else:
                    ElasticLogger().write_log('INFO', "Job finished successfully", flatten_json(proc), push_bulk=True,print_log=True)
                    proc['log_handle'].close()

                    # ack
                    #channel.basic_ack(proc['job_tag'])
                    proc_running.remove(proc)
                    break


        ### Reading one job from queue
        method_frame, properties, body = channel.basic_get(args.channel)

        if body is not None:

            # Display the message parts
            body = json.loads(body.decode())
            print(method_frame)
            print(properties)
            print(body)

            if body['command'] == 'kill job':
                print('kill job!')
                try:
                    pid_to_kill = [proc['pid'] for proc in proc_running if proc['experiment_name'] == body]
                    body = 'kill ' + str(pid_to_kill[0])
                except:
                    ElasticLogger().write_log('INFO', "job runner exception", {'error_message': traceback.format_exc()},
                                              push_bulk=True, print_log=True)
                    channel.basic_ack(method_frame.delivery_tag)
                    time.sleep(2)
            elif body['command'] == 'train':
                bash_command = 'python -m allennlp.run train ' + body['master_config']
                bash_command += '--s ' + body['model_dir'] + properties.headers['name'] + ' '
                # Building the python command with arguments
                bash_command += '-o "' + str(body['override_config']).replace('True', 'true').replace('False', 'false') + '"'
                bash_command == body['include-package']

            if args.shell == 'bash':
                bash_command = 'nohup ' + bash_command + ' &'
            else:
                bash_command = 'nohup ' + bash_command + ' &'

            # Creating the log dir
            log_file = 'logs/' + properties.headers['name'] + '.txt'
            log_dir_part = log_file.split('/')
            for dir_depth in range(len(log_dir_part)-1):
                if not os.path.isdir('/'.join(log_dir_part[0:dir_depth+1])):
                    os.mkdir('/'.join(log_dir_part[0:dir_depth+1]))

            # performing git pull before each execution
            Popen("git pull origin master", shell=True, preexec_fn=os.setsid)

            # Executing
            print(bash_command)
            with open(log_file,'wb') as f:

                # TODO run bash command
                # bash_command
                wa_proc = Popen("nohup python dummy_job.py &", shell=True, preexec_fn=os.setsid,stdout=f,stderr=f)

            # open log file for reading
            proc_running.append({'job_tag':method_frame.delivery_tag,'config':body, 'command':bash_command, \
                                 'log_file':log_file,'log_handle':open(log_file,'r'), 'log_snapshot':'',\
                                 'experiment_name':properties.headers['name'], 'alive': True,\
                                 'pid': wa_proc.pid+1, 'start_time': time.time()})
            # we are not persistant for now ...
            channel.basic_ack(method_frame.delivery_tag)
            ElasticLogger().write_log('INFO', "Job Started", flatten_json(proc), push_bulk=True, print_log=True)
            time.sleep(2)



        if iter_count % 10 == 1:
            # Virtual memory usage
            # Giving info on the processes running and GPU status

            # resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            for proc in proc_running:
                try:
                    proc['memory'] = psutil.Process(proc['pid']).memory_info()
                    #proc['command'] = ' '.join(psutil.Process(wa_proc.pid+1).cmdline())
                except:
                    proc['alive'] = False

                ElasticLogger().write_log('INFO', "Job Status", {'experiment_name':proc['experiment_name'],
                            'log_snapshot':proc['log_snapshot']}, push_bulk=True, print_log=True)

            try:
                gpu_mem = gpu_memory_mb()
                # Ugly patch for misconfigured GPUs...
                if args.channel == 'rack-jonathan-g02':
                    gpu_mem = {(3 - key):val for key,val in gpu_mem.items()}
                free_gpus = [i for i, gpu in enumerate(gpu_mem.keys()) if gpu_mem[gpu] < 1700]
            except:
                free_gpus = []

            ElasticLogger().write_log('INFO', "Machine Status", {'gpus':gpu_mem,'free_gpus':free_gpus,\
                                                      'num_procs_running':len(proc_running),}, push_bulk=True,print_log=True)

        time.sleep(6)


    except:
        # something went wrong
        time.sleep(3)
        print(traceback.format_exc())
        ElasticLogger().write_log('INFO', "job runner exception", {'error_message': traceback.format_exc()},
                                  push_bulk=True, print_log=True)

        # maybe this is a connection error:
        try:
            channel.close()
            connection.close()
        except:
            print('close failed')

        try:
            connection_params = pika.URLParameters(
                'amqp://imfgrmdk:Xv_s9oF_pDdrd0LlF0k6ogGBOqzewbqU@barnacle.rmq.cloudamqp.com/imfgrmdk')
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
            print('reconnected')
        except:
            print('reconnecting failed')



channel.close()
connection.close()
