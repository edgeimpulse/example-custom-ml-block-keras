const EdgeImpulseApi = require('edge-impulse-api').EdgeImpulseApi;
const spawn = require('child_process').spawn;
const Path = require('node:path');
const fs = require('node:fs');
const program = require('commander');

const packageVersion = JSON.parse(fs.readFileSync(Path.join(__dirname, 'package.json'), 'utf-8')).version;
const API_KEY_FILE = Path.join(__dirname, '..', '.ei-api-key');
const EI_BLOCK_CONFIG = Path.join(__dirname, '..', '.ei-block-config');

if (!fs.existsSync(API_KEY_FILE)) {
    console.log(`Missing ${API_KEY_FILE}`);
    process.exit(1);
}
if (!fs.existsSync(EI_BLOCK_CONFIG)) {
    console.log(`Missing ${EI_BLOCK_CONFIG}`);
    process.exit(1);
}

const API_KEY = (fs.readFileSync(Path.join(API_KEY_FILE), 'utf-8')).trim();
if (!API_KEY.startsWith('ei_')) {
    console.log(`API Key (in .ei-api-key) does not start with "ei_"`);
    process.exit(1);
}

program
    .description('Verify blocks in Edge Impulse')
    .version(packageVersion)
    .option('--push-block', 'Push the block to Edge Impulse')
    .option('--impulse-id <impulseId>', 'If set, selects a specific impulse')
    .option('--learn-id <learnId>', 'If set, selects a specific learn block')
    .allowUnknownOption(false)
    .parse(process.argv);

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    try {
        const pushBlock = !!program.pushBlock;
        const impulseId = program.impulseId ? Number(program.impulseId) : undefined;
        const learnId = program.learnId ? Number(program.learnId) : undefined;

        const blockConfig = JSON.parse(await fs.promises.readFile(EI_BLOCK_CONFIG, 'utf-8'));
        if (blockConfig.version !== 2) {
            throw new Error(`${EI_BLOCK_CONFIG}, version is not "2" but "${blockConfig.version}"`);
        }
        if (!(blockConfig.config || { })['edgeimpulse.com']) {
            throw new Error(`${EI_BLOCK_CONFIG}, missing "config[edgeimpulse.com]". Did you run 'edge-impulse-blocks init'`);
        }
        if (!blockConfig.config['edgeimpulse.com']['organizationId']) {
            console.log(blockConfig.config['edgeimpulse.com']);
            throw new Error(`${EI_BLOCK_CONFIG}, missing "config[edgeimpulse.com][organizationId]". Did you run 'edge-impulse-blocks init'`);
        }
        if (!blockConfig.config['edgeimpulse.com']['id']) {
            throw new Error(`${EI_BLOCK_CONFIG}, missing "config[edgeimpulse.com][id]". Did you run 'edge-impulse-blocks push'`);
        }

        if (pushBlock) {
            console.log('Pushing block...')
            await spawnHelper('edge-impulse-blocks', ['push'], { cwd: Path.join(__dirname, '..') });
            console.log('Pushing block OK');
            console.log('');
        }

        const api = new EdgeImpulseApi();
        await api.authenticate({
            method: 'apiKey',
            apiKey: API_KEY,
        });

        // list all projects (if you authenticate via API you just get one, the project for which you have the API key)
        let project = (await api.projects.listProjects()).projects[0];
        console.log('Project:', project.owner, '/', project.name);

        // select impulse / learn block
        let impulseRes = await api.impulse.getAllImpulses(project.id);

        let impulse;
        if (impulseId) {
            impulse = impulseRes.impulses.find(x => x.id === impulseId);
            if (!impulse) {
                throw new Error(`Could not find impulse with ID ${impulseId} (from --impulse-id)`);
            }
        }
        else {
            if (impulseRes.impulses.length === 0) {
                throw new Error(`This project has no impulses`);
            }
            if (impulseRes.impulses.length > 1) {
                throw new Error(`This project has multiple impulses (${impulseRes.impulses.map(x => `${x.name} (ID: ${x.id})`).join(', ')}). ` +
                    `Specify the impulse via --impulse-id <id>`);
            }
            impulse = impulseRes.impulses[0];
        }

        let learnBlock;
        if (learnId) {
            learnBlock = impulse.learnBlocks.find(x => x.id === learnId);
            if (!learnBlock) {
                throw new Error(`Could not find learn block with ID ${learnId} (from --learn-id)`);
            }
        }
        else {
            if (impulse.learnBlocks.length === 0) {
                throw new Error(`Impulse has no learn blocks`);
            }
            if (impulse.learnBlocks.length > 1) {
                throw new Error(`This impulse has multiple learn blocks (${impulse.learnBlocks.map(x => `${x.name} (ID: ${x.id})`).join(', ')}). ` +
                    `Specify the learn block via --learn-id <id>`);
            }
            learnBlock = impulse.learnBlocks[0];
        }

        // and retrain with same config
        let trainJob = await api.jobs.trainKerasJob(project.id, learnBlock.id, {
            "trainTestSplit": 0.2,
            "customValidationMetadataKey": "",
            "autoClassWeights": false,
            "profileInt8": true,
            "mode": "visual",
            "visualLayers": [{
                "type": "transfer_organization",
                "organizationModelId": blockConfig.config['edgeimpulse.com']['id'],
            }],
            "augmentationPolicyImage": "none",
            "useLearnedOptimizer": false,
            "blockParameters": {},
            "customParameters": {   // <-- maps to parameters.json
                "epochs": "30",
                "learning-rate": "0.001",
            },
        });
        console.log('Created train job with ID', trainJob.id);

        await api.runJobUntilCompletion({
            type: 'project',
            projectId: project.id,
            jobId: trainJob.id,
        }, data => {
            process.stdout.write(data);
        });

        console.log('Train job completed');
    }
    catch (ex) {
        console.log('Failed to make a request', ex);
        process.exit(1);
    }
})();


function spawnHelper(
    command,
    args,
    opts
) {
    return new Promise((resolve, reject) => {
        const p = spawn(command, args, { env: process.env, cwd: opts.cwd });

        const allData = [];

        p.stdout.on('data', (data) => {
            process.stdout.write(data);
            allData.push(data);
            if (
                opts.readySignal &&
                data.toString().includes(opts.readySignal)
            ) {
                resolve(Buffer.concat(allData).toString('utf-8'));
            }
        });

        p.stderr.on('data', (data) => {
            process.stderr.write(data);
            allData.push(data);
        });

        p.on('error', () => {
            reject(
                'Error running command: ' +
                    Buffer.concat(allData).toString('utf-8')
            );
        });

        p.on('close', (code) => {
            if (code === 0 || opts.ignoreErrors === true) {
                resolve(Buffer.concat(allData).toString('utf-8'));
            } else {
                reject(
                    'Error code was not 0: ' +
                        Buffer.concat(allData).toString('utf-8')
                );
            }
        });
    });
}
