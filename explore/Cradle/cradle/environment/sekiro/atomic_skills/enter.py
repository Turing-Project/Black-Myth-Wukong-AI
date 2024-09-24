from cradle.gameio.io_env import IOEnvironment
from cradle.environment.sekiro.skill_registry import register_skill

io_env = IOEnvironment()



@register_skill("enter_the_temple")
def enter_the_temple():
    io_env.key_press('E')
    
@register_skill("take_a_rest")
def take_a_rest():
    io_env.key_press('E')
__all__ = []
