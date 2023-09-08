from datetime import datetime


class Logger(object):

    def __init__(self, logfile_path):
        super(Logger, self).__init__()
        if logfile_path is None:
            self.logfile_path = None
        else:
            self.logfile_path = logfile_path / ('run_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.txt')

    def write(self, text, print_text=True):
        if print_text:
            print(text)
        if self.logfile_path is not None:
            with open(self.logfile_path, mode='a') as logfile:
                logfile.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ': ' + text + '\n')
