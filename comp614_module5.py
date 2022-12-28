"""
COMP 614
Provided code for homework 5. Includes an index of all files found in the 
wikipedia_articles directory.
"""

ALL_FILES = ['elizabeth_i.xml', 'detroit_tigers.xml', 'korean_war.xml', 'fascism.xml', 'liverpool.xml',
             'confederate_states_of_america.xml', 'vincent_van_gogh.xml', 'lithuania.xml',
             'heavy_metal_music.xml', 'charlotte_bronte.xml', 'columbia_university.xml',
             'central_intelligence_agency.xml', 'communism.xml', 'chemistry.xml', 'led_zeppelin.xml',
             'homeopathy.xml', 'hindusim.xml', 'claude_monet.xml', 'detroit.xml', 'elizabeth_ii.xml',
             'history_of_israel.xml', 'mary,_queen_of_scots.xml', 'richard_nixon.xml', 'sculpture.xml',
             'manhattan_project.xml', 'mariah_carey.xml', 'history_of_france.xml', 'country_music.xml',
             'new_york_yankees.xml', 'ronald_reagan.xml', 'guns_n_roses.xml', 'dna.xml',
             'roman_republic.xml', 'matt_damon.xml', 'architecture.xml', 'james_stewart.xml',
             'green_day.xml', 'maya_angelou.xml', 'pink_singer.xml', 'florence_griffifth_joyner.xml',
             'metallica.xml', 'california.xml', 'aristotle.xml', 'serena_williams.xml', 'texas.xml',
             'trinity_university.xml', 'houston.xml', 'hip_hop_music.xml', 'lebron_james.xml',
             'jimi_hendrix.xml', 'beyonce.xml', 'joseph_stalin.xml', 'rudy_giuliani.xml',
             'alfred_hitchcock.xml', 'university_of_michigan.xml', 'paris.xml',
             'global_warming_controversy.xml', 'new_jersey.xml', 'arnold_schwarzenegger.xml',
             'history_of_the_netherlands.xml', 'pope_john_paul_ii.xml', 'marie_antoinette.xml',
             'al_qaeda.xml', 'italy.xml', 'european_union.xml', 'history_of_islam.xml',
             'mark_twain.xml', 'gaza_strip.xml', 'radiation_therapy.xml', 'rock_music.xml',
             'bipolar_disorder.xml', 'immanuel_kant.xml', 'major_depressive_disorder.xml',
             'friedrich_neitzsche.xml', 'music.xml', 'martin_scorsese.xml', 'jazz.xml',
             'frida_kahlo.xml', 'cardiovascular_disease.xml', 'alternative_rock.xml',
             'netherlands.xml', 'dwight_d_eisenhower.xml', 'afghanistan.xml', 'reincarnation.xml',
             'history_of_painting.xml', 'abolitionism.xml', 'jennifer_lopez.xml', 'art.xml',
             'university_of_southern_california.xml', 'michael_phelps.xml', 'slovenia.xml',
             'orson_welles.xml', 'cancer.xml', 'frank_sinatra.xml', 'nuclear_power.xml',
             'noam_chomsky.xml', 'houston_astros.xml', 'alex_puccio.xml', 'india.xml', 'romania.xml',
             'britney_spears.xml', 'the_beatles.xml', 'diana_princess_of_wales.xml', 'bill_clinton.xml',
             'greece.xml', 'spain.xml', 'russia.xml', 'john_cho.xml', 'pope_pius_xii.xml',
             'neoplasm.xml', 'martin_luther_king_jr.xml', 'new_york_city.xml', 'babe_ruth.xml',
             'leonardo_da_vinci.xml', 'elvis_presley.xml', 'baltimore_orioles.xml', 'austin,_texas.xml',
             'human_evolution.xml', 'brazil.xml', 'spice_girls.xml', 'herman_melville.xml',
             'david_bowie.xml', 'pop_music.xml', 'ghana.xml', 'serbia.xml', 'atlanta_braves.xml',
             'electronic_dance_music.xml', 'olympic_games.xml', 'history_of_europe.xml',
             'yale_university.xml', 'spanish_language.xml', 'empire_state_building.xml',
             'national_security_agency.xml', 'rome.xml', 'george_h._w._bush.xml', 'henry_viii.xml',
             'ecology.xml', 'adam_ondra.xml', 'emma_watson.xml', 'keanu_reeves.xml',
             'rice_university.xml', 'kurt_vonnegut.xml', 'michaelangelo.xml', 'soviet_union.xml',
             'capitalism.xml', 'abraham_lincoln.xml', 'jackie_robinson.xml', 'mahatma_gandhi.xml',
             'sigmund_freud.xml', 'on_the_origin_of_species.xml', 'thomas_jefferson.xml',
             'french_revolution.xml', 'charles_darwin.xml', 'montana.xml', 'university_of_florida.xml',
             'nuclear_winter.xml', 'andrew_jackson.xml', 'american_revolutionary_war.xml',
             'meryl_streep.xml', 'toni_morrison.xml', 'plato.xml', 'kobe_bryant.xml', 'usain_bolt.xml',
             'natalie_coughlin.xml', 'china.xml', 'jimmy_carter.xml', 'roger_federer.xml',
             'natural_selection.xml', 'sabah.xml', 'jerusalem.xml', 'lynyrd_skynyrd.xml',
             'maroon_5.xml', 'black_sabbath.xml', 'portugal.xml', 'chemotherapy.xml',
             'audrey_hepburn.xml', 'benign_tumor.xml', 'william_faulkner.xml', 'leonardo_dicaprio.xml',
             'england.xml', 'general_relativity.xml', 'mumbai.xml', 'slavery.xml', 'akiyo_noguchi.xml',
             'pakistan.xml', 'harvard_university.xml', 'colombia.xml', 'margaret_thatcher.xml',
             'lupita_nyongo.xml', 'punk_rock.xml', 'johnny_depp.xml', 'norway.xml', 'sandra_oh.xml',
             'television.xml', 'venus_williams.xml', 'chicago.xml', 'jane_austen.xml', 'poland.xml',
             'michael_jordan.xml', 'biology.xml', 'william_shakespeare.xml', 'marilyn_manson.xml',
             'barack_obama.xml', 'sue_bird.xml', 'ottoman_empire.xml', 'tiger_woods.xml',
             'virginia_woolf.xml', 'philippines.xml', 'emily_bronte.xml', 'kylie_minogue.xml',
             'evolution.xml', 'roman_empire.xml', 'history_of_poland.xml', 'sanskrit.xml',
             'boston_red_sox.xml', 'cuban_missile_crisis.xml', 'washington_dc.xml', 'pottery.xml',
             'federal_bureau_of_investigation.xml', 'stroke.xml', 'george_washington.xml',
             'janja_garnbret.xml', 'morgan_freeman.xml', 'harvey_mudd_college.xml',
             'american_civil_war.xml', 'iran.xml', 'charles_dickens.xml', 'queen.xml',
             'black_metal.xml', 'ac_dc.xml', 'simone_biles.xml', 'charlie_chaplin.xml']

# Prepend the appropriate directory to each filename
ALL_FILES = ["wikipedia_articles/" + filename for filename in ALL_FILES]