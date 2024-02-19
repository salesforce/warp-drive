import torch
from warp_drive.managers.data_manager import CUDADataManager


class RingBuffer:
    """
    We manage the batch data as a circular queue
    """

    def __init__(
        self,
        name: str = None,
        size: int = None,
        data_manager: CUDADataManager = None,
    ):
        self.buffer_name = f"RingBuffer_{name}"
        assert data_manager.is_data_on_device_via_torch(name)
        # initializing queue with none
        self.front = -1
        self.rear = -1
        self.current_size = 0

        self.queue = data_manager.data_on_device_via_torch(name=name)
        if size is None:
            self.size = data_manager.get_shape(name)[0]
        else:
            self.size = size
        assert self.size <= data_manager.get_shape(name)[0], \
            f"The managed the ring buffer size could not exceed the size of the container: {name}"

    def enqueue(self, data):
        assert isinstance(data, torch.Tensor)
        # condition if queue is full
        if (self.rear + 1) % self.size == self.front:
            self._dequeue()
        # condition for empty queue
        if self.front == -1:
            self.front = 0
            self.rear = 0
            self.queue[self.rear] = data
        else:
            # next position of rear
            self.rear = (self.rear + 1) % self.size
            self.queue[self.rear] = data
        self.current_size += 1

    def _dequeue(self):
        if self.front == -1:
            return
        # condition for only one element
        elif self.front == self.rear:
            self.front = -1
            self.rear = -1
        else:
            self.front = (self.front + 1) % self.size
        self.current_size -= 1

    def unroll(self):
        # we unroll the circular queue to a flattened array with index following the order from front to tail
        if self.front == -1:
            return None

        elif self.rear >= self.front:
            return self.queue[self.front: self.rear+1]

        else:
            return torch.roll(self.queue, shifts=-self.front, dims=0)[:self.current_size]

    def isfull(self):
        return self.current_size == self.size


class RingBufferManager(dict):

    def add(self, name, size=None, data_manager=None):
        r = RingBuffer(name=name, size=size, data_manager=data_manager)
        self[name] = r

    def get(self, name):
        assert name in self, \
            f"{name} not in the RingBufferManager"
        return self[name]

    def has(self, name):
        return name in self


