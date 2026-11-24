"""Entry point: python -m imatrix <generate|analyze|recommend|quantize|validate|compare|correlate|evaluate> [options]"""
import sys

MODULES={
 'generate':('generate',None),
 'analyze':('research','analyze'),
 'recommend':('research','recommend'),
 'mixed-quantize':('research','quantize'),
 'validate':('research','validate'),
 'compare':('research','compare'),
 'quantize':('quantize_runs',None),
 'correlate':('correlations',None),
 'evaluate':('evaluate',None),
}

def main():
    if len(sys.argv)<2 or sys.argv[1] in ('--help','-h'):
        print(__doc__);print('Commands: '+', '.join(MODULES));return
    operation=sys.argv.pop(1)
    if operation not in MODULES:raise SystemExit(f'Unknown command {operation}. Available: {list(MODULES)}')
    module,argument=MODULES[operation]
    if argument:sys.argv.insert(1,argument)
    from importlib import import_module
    import_module(f'.{module}',__package__).main()
if __name__=='__main__':main()
