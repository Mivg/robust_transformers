from dataclasses import dataclass, field
import json
from attacks.strategies import get_strategy_names


@dataclass
class AdversarialTrainingArguments:
    """
    Arguments needed to train an adversarial-trained model
    """
    orig_lambda: float = field(
        default=1.,
        metadata={
            "help": "A float in [0,1] to control how much weight to put on the loss coming from the original input compared to "
                    "the adversarial attack input"
        }
    )
    syn_task: str = field(
        default=None,
        metadata={
            "help": 'The task traing with online adversarial training (our of boolq, sst2 and imdb)'
        }
    )

    adversarial_every: int = field(
        default=1,
        metadata={
            "help": "every how many batches perform adversarial training (cumulative batches counts together thus it will "
                    "count as what's given here times gradient accumulation steps, i.e. every 1 means alwas, every 2 will be 1 yes, 1 no, "
                    "or if gradient_accumulation is 4 it will be 4 steps yes, then 4 steps no)"
        }
    )
    num_attacks: int = field(
        default=50,
        metadata={
            "help": "number of forward passes to make in the attack exploration phase"
        }
    )
    n_exploits: int = field(
        default=1,
        metadata={
            "help": "number of exploitation to perform in each adversarial training batch"
        }
    )
    async_adv_batches: bool = field(
        default=False,
        metadata={
            "help": "If set, adversarial batches are samples seperatley from regular batches"
        }
    )
    n_adv_loaders: int = field(
        default=1,
        metadata={
            "help": "number of adversearial loaders to use if in async_adv_batches mode"
        }
    )
    pert_text: str = field(
        default='Auto',
        metadata={
            "help": "every how many batches perform adversarial training (cumulative batches counts together thus it will. "
                    "one of ['a', 'b', 'Auto', 'random']"
        }
    )
    strategy_name: str = field(
        default='rand_synonym_swap',
        metadata={
            "help": f"every how many batches perform adversarial training (cumulative batches counts together thus it will. "
            f"one of: {get_strategy_names()}"
        }
    )
    strategy_params: str = field(
        default='{}',
        metadata={
            "help": "ll parameters needed for the creation of the strategy"
            # Example: {\"synonyms_path\":\"../data/counterfitted_neighbors.json\",\"edit_distance\":5}
        }
    )
    tfhub_cache_dir: str = field(
        default='/media/disk2/maori/tfhub_cache',
        metadata={
            "help": "path to which tensorflow hub should cache models"
        }
    )

    def __post_init__(self):
        self.strategy_params = json.loads(self.strategy_params)
        if self.strategy_name not in get_strategy_names():
            raise ValueError(f'Strategy params must be one of {get_strategy_names()}, got {self.strategy_name}')
        if self.pert_text not in ['a', 'b', 'Auto', 'random']:
            raise ValueError(f"Strategy params must be one of {['a', 'b', 'Auto', 'random']}, got {self.pert_text}")
