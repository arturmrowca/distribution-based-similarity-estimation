from abc import ABCMeta, abstractmethod

class BaseNode(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Initializes the PipelineNode using the given parameters
        :param parameters: 
        """

    def _pre_run_checks(self, data):
        """
        This method should not be called manually. It is must be only called from run of the parent (PipelineNode)
        class
        :param data: the data to check
        :return: 
        """
        return

    @abstractmethod
    def run(self, data):
        """
        Runs the nodes functionality on the data and returns the data dictionary, including the new, generated fields
        as well as a metrics dictionary, holding eventual results.
        :param data: the data to do the calculation
        :return: tuple of (enhanced) dataset and the metrics
        """

        self._pre_run_checks(data=data)

