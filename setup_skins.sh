#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd Minecraft
curl https://people.eecs.berkeley.edu/~cassidy_laidlaw/CustomSkinLoader_ForgeLegacy-14.13-SNAPSHOT-317.jar -o run/mods/CustomSkinLoader_ForgeLegacy-14.13-SNAPSHOT-317.jar

SKINS_DIR=$SCRIPT_DIR/data/minecraft/skins
MOD_SKINS_DIR=run/CustomSkinLoader/LocalSkin/skins
mkdir -p $MOD_SKINS_DIR

cp $SKINS_DIR/steve.png $MOD_SKINS_DIR/ppo.png
cp $SKINS_DIR/steve.png $MOD_SKINS_DIR/ppo_0.png
cp $SKINS_DIR/qt_robot.png $MOD_SKINS_DIR/ppo_1.png
