import torch

from funcodec.train.distributed_utils import DistributedOption
from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from funcodec.iterators.sequence_iter_factory import SequenceIterFactory



def init_sequence_iter_factory(args, rank, mode) -> SequenceIterFactory:
    distributed_option = DistributedOption()
    distributed_option.distributed = True
    distributed_option.dist_rank = rank
    distributed_option.local_rank = rank

    distributed_option.dist_world_size = torch.distributed.get_world_size()
    iter_option = Text2AudioGenTask.build_iter_options(args, distributed_option, mode)
    return Text2AudioGenTask.build_sequence_iter_factory(args, iter_option, mode)


def init_dm_sequence_iter_factory(args, rank, mode) -> SequenceIterFactory:
    """
    Dynamic Sequence Iterator Factory
    If args.preprocess is speficified, we use the preprocess_fn
    """
    distributed_option = DistributedOption()
    distributed_option.distributed = True
    distributed_option.dist_rank = rank
    distributed_option.local_rank = rank

    distributed_option.dist_world_size = torch.distributed.get_world_size()
    iter_option = Text2AudioGenTask.build_iter_options(args, distributed_option, mode)
    # iter_option.preprocess_fn = 
    from _funcodec_data.build_sequence_iter import build_sequence_iter_factory
    return build_sequence_iter_factory(args, iter_option, mode)
