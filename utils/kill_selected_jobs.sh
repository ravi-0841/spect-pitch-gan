#!/bin/bash
for i in {$1..$2..1};do scancel $i; done
