#!/bin/bash
#$ -cwd 
#$ -j y 
#$ -S /bin/bash

# Copyright 2015   David Snyder
#           2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#           2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.
#
# This example script shows how to replace the GMM-UBM
# with a DNN trained for ASR. It also demonstrates the 
# using the DNN to create a supervised-GMM.

#. ./cmd.sh
#. ./path.sh
#set -e
#
#if [ $# != 1 ] ; then 
#	echo "USAGE: $0 wav_path" 
#	echo " e.g.: $0 ./wav" 
#	exit 1;
#fi 
#
#if [ -d "./data" ];then
#	rm -rf ./data
#fi
#
##wavdir=`pwd`/wav
#wavdir=$1
#datadir=`pwd`/data
#logdir=`pwd`/data/log
#featdir=`pwd`/data/feat
#
#. parse_options.sh || exit 1;
#
#DataPre=1
#FIXDATA=1
#FeatureForMfcc=1
#VAD=1
#EXTRACT=1
#
#if [ $DataPre -eq 1 ]; then
#	echo ==========================================
#	echo "get utt2spk, DataPre start on" `date`
#	echo ==========================================
#
#	python make_data.py $wavdir $datadir
#	utils/utt2spk_to_spk2utt.pl $datadir/utt2spk > $datadir/spk2utt || exit 1
#	utils/spk2utt_to_utt2spk.pl $datadir/spk2utt > $datadir/utt2spk || exit 1
#
#	echo ===== data preparatin finished successfully `date`==========
#fi
#
#
#if [ $FIXDATA -eq 1 ]; then
#    echo ==========================================
#	echo "sorted spk2utt ... : fix_data_dir start on" `date`
#	echo ==========================================
#    utils/fix_data_dir.sh $datadir
#	echo ====== fix_data_dir finished successfully `date` ==========
# fi
#
#
#if [ $FeatureForMfcc -eq 1 ]; then
#	 echo ==========================================
#	 echo "FeatureForSpeaker start on" `date`
#	 echo ========================================== 
#	# Extract speaker features MFCC.
#    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 1 --cmd "$train_cmd" \
#    $datadir $logdir/make_enrollmfcc $featdir/mfcc
#	echo ==== FeatureForSpeaker test successfully `date` ===========
#
#fi 
#
#
#if [ $VAD -eq 1 ];then
#	 echo ==========================================
#	 echo "generate vad file in data/train, VAD start on" `date`
#	 echo ==========================================
#	# Compute VAD decisions. These will be shared across both sets of features.
#	sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
#    $datadir $logdir/make_enrollvad $featdir/vad
#
#	utils/fix_data_dir.sh $datadir
#	
#echo ========== VAD test successfully `date` ===============
#fi
#
#
#if [ $EXTRACT -eq 1 ]; then
#	 echo ==========================================
#	 echo "EXTRACT start on" `date`
#	 echo ==========================================
#	# Extract the iVectors
#	sid/extract_ivectors.sh --cmd "$train_cmd" --nj 1 \
#	   model_3000h $datadir $featdir/ivectors_enroll_mfcc
#	   
#	echo ========= EXTRACT just for testing `date`=============
#fi   


. ./cmd.sh
. ./path.sh
set -e

if [ $# != 1 ] ; then
    echo "USAGE: $0 wav_path"
    echo " e.g.: $0 ./wav"
    exit 1;
fi

if [ -d "./data" ];then
    rm -rf ./data
fi

#wavdir=`pwd`/wav
wavdir=$1
datadir=`pwd`/data
logdir=`pwd`/data/log
featdir=`pwd`/data/feat

. parse_options.sh || exit 1;

DataPre=1
FIXDATA=1
FeatureForMfcc=1
VAD=1
TRAIN_UBM=1
TRAIN_IVECTOR_EXTRACTOR=1
EXTRACT=1


if [ $DataPre -eq 1 ]; then
    echo ==========================================
    echo "get utt2spk, DataPre start on" `date`
    echo ==========================================

    python make_data.py $wavdir $datadir
    utils/utt2spk_to_spk2utt.pl $datadir/utt2spk > $datadir/spk2utt || exit 1
    utils/spk2utt_to_utt2spk.pl $datadir/spk2utt > $datadir/utt2spk || exit 1

    echo ===== data preparatin finished successfully `date`==========
fi


if [ $FIXDATA -eq 1 ]; then
    echo ==========================================
    echo "sorted spk2utt ... : fix_data_dir start on" `date`
    echo ==========================================
    utils/fix_data_dir.sh $datadir
    echo ====== fix_data_dir finished successfully `date` ==========
 fi


if [ $FeatureForMfcc -eq 1 ]; then
     echo ==========================================
     echo "FeatureForSpeaker start on" `date`
     echo ==========================================
    # Extract speaker features MFCC.
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 1 --cmd "$train_cmd" \
    $datadir $logdir/make_enrollmfcc $featdir/mfcc
    echo ==== FeatureForSpeaker test successfully `date` ===========

fi


if [ $VAD -eq 1 ];then
     echo ==========================================
     echo "generate vad file in data/train, VAD start on" `date`
     echo ==========================================
    # Compute VAD decisions. These will be shared across both sets of features.
    sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
    $datadir $logdir/make_enrollvad $featdir/vad

    utils/fix_data_dir.sh $datadir
    
echo ========== VAD test successfully `date` ===============
fi

#add
if [ $TRAIN_UBM -eq 1 ]; then
    echo ==========================================
    echo "TRAIN_UBM start on" `date`
    echo ==========================================
    num_components=512

    sid/train_diag_ubm.sh --nj 1 --cmd "$train_cmd" \
      $datadir $num_components exp/diag_ubm

    sid/train_full_ubm.sh --nj 1 --cmd "$train_cmd" \
      $datadir exp/diag_ubm exp/full_ubm

    echo ========== TRAIN_UBM finished successfully `date` ===============
fi

if [ $TRAIN_IVECTOR_EXTRACTOR -eq 1 ]; then
    echo ==========================================
    echo "TRAIN_IVECTOR_EXTRACTOR start on" `date`
    echo ==========================================
    ivector_dim=400
    sid/train_ivector_extractor.sh --nj 1 --cmd "$train_cmd" \
      --num-iters 5 exp/full_ubm/final.ubm $datadir exp/extractor

    echo ========== TRAIN_IVECTOR_EXTRACTOR finished successfully `date` ===============
fi

if [ $EXTRACT -eq 1 ]; then
    echo ==========================================
    echo "EXTRACT start on" `date`
    echo ==========================================
    # Extract the iVectors
    sid/extract_ivectors.sh --cmd "$train_cmd" --nj 1 \
       exp/extractor $datadir $featdir/ivectors_enroll_mfcc
       
    echo ========= EXTRACT finished successfully `date`=============
fi

