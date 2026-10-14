import sys; sys.path.insert(0,'/home/claude/rd')
import produce
RED = dict(base=(176, 10, 18), hi=(255, 150, 160), lo=(60, 0, 6), dark=(40, 0, 4), edge=(80, 0, 8), stitch=(255, 205, 210), dots=(255, 190, 196))
produce.curved('SHEA', border=RED, tag='_RED-METALLIC')
produce.straight('SHEA', border=RED, tag='_RED-METALLIC')
