#ifndef PLAN_FRAME_SEQUENCE_H
#define PLAN_FRAME_SEQUENCE_H

#include "opencv2/core.hpp"
#include <vector>
#include <string>

std::vector<int> PlanFrameSequence(const std::string& basePath,
    int firstIndex,
    int lastIndex,
    int maxStep);

std::vector<int> ComputeReconstructionSequence(const std::vector<int>& sparseSeq,
    const std::vector<int>& frameSeq);

std::vector<int> ReadFrameSequence(const std::string& filepath);

#endif // PLAN_FRAME_SEQUENCE_H