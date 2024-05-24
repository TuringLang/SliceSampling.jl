
module SliceSamplingTuringExt

if isdefined(Base, :get_extension)
    using SliceSampling: Transition
    using Turing: Turing
else
    using ..SliceSampling: Transition
    using ..Turing: Turing
end

Turing.Inference.getparams(::Turing.DynamicPPL.Model, sample::Transition) = sample.params

end
