import os
from keras.utils.generic_utils import Progbar
from six.moves.urllib.request import urlretrieve
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError

class ProgressTracker(object):
        # Maintain progbar for the lifetime of download.
        # This design was chosen for Python 2.7 compatibility.
        progbar = None

def dl_progress(count, block_size, total_size):
    if ProgressTracker.progbar is None:
        if total_size is -1:
            total_size = None
        ProgressTracker.progbar = Progbar(total_size)
    else:
        ProgressTracker.progbar.update(count * block_size)
        
def download(url, savepath):
    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(url, savepath, dl_progress)
        except URLError as e:
            raise Exception(error_msg.format(url, e.errno, e.reason))
        except HTTPError as e:
            raise Exception(error_msg.format(url, e.code, e.msg))
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(savepath):
            os.remove(savepath)
        raise
    ProgressTracker.progbar = None