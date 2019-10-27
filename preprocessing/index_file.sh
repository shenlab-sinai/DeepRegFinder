#!/bin/bash

bgzip -c $1 > $2
tabix -p bed $2