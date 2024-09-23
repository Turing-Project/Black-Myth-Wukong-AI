import pydirectinput
import time


class walkToBoss:
        '''Walk to boss class - hard coded paths from the bonfire to the boss'''

        '''Constructor'''
        def __init__(self, BOSS):
                self.BOSS = BOSS        #Boss number | 99/100 reserved for PVP


        '''Walk to boss function'''
        def perform(self):
                '''PVE'''
                if self.BOSS == 1:
                        self.boss1()
                # elif self.BOSS == 2:
                #         self.boss2()
                # elif self.BOSS == 3:
                #         self.boss3()
                # elif self.BOSS == 4:
                #         self.boss4()
                # elif self.BOSS == 5:
                #         self.boss5()
                # elif self.BOSS == 6:
                #         self.boss6()
                # elif self.BOSS == 7:
                #         self.boss7()
                # elif self.BOSS == 8:
                #         self.boss8()
                # elif self.BOSS == 9:
                #         self.boss9()
                # elif self.BOSS == 10:
                #         self.boss10()
                # elif self.BOSS == 11:






        '''1 Margit, The fell Omen'''
        def boss1(self):
                pydirectinput.press('e')
                time.sleep(3)
                pydirectinput.press('esc')
                time.sleep(1)
                pydirectinput.keyDown('shift')
                pydirectinput.keyDown('a')
                time.sleep(0.1)
                pydirectinput.keyUp('a')
                pydirectinput.keyDown('w')
                time.sleep(4.3)
                pydirectinput.keyUp('w')
                pydirectinput.keyDown('d')
                time.sleep(0.1)
                pydirectinput.keyUp('d')
                pydirectinput.keyDown('w')
                time.sleep(5)
                pydirectinput.keyDown('d')
                time.sleep(1.3)
                pydirectinput.keyUp('d')
                time.sleep(3)
                pydirectinput.keyDown('d')
                time.sleep(6)
                pydirectinput.keyUp('d')
                time.sleep(2)
                pydirectinput.keyUp('w')
                pydirectinput.press('k')
                pydirectinput.keyUp('shift')

#Run the function to test it
def test():
    print("👉👹 3")
    time.sleep(1)
    print("👉👹 2")
    time.sleep(1)
    print("👉👹 1")
    time.sleep(1)
    walkToBoss(1).boss1()
if __name__ == "__main__":
    test()
